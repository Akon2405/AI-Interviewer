import sounddevice as sd
import numpy as np
import queue
import whisper
import pyttsx3
import scipy.io.wavfile as wav
import torch
from silero_vad import load_silero_vad

# Import the brain module
from brain import InterviewerBrain

# 1. SETUP
audio_queue = queue.Queue()
samplerate = 16000
device_id = 4  # Your Mic ID

print("🧠 Loading Whisper (Small) & Silero VAD models...")
whisper_model = whisper.load_model("small")
vad_model = load_silero_vad()

def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy())

# 2. THE MAIN LOOP
def start_listening():
    print("🧠 Initializing Interviewer Brain...")
    brain = InterviewerBrain()
    brain.start_thinking_thread()
 
    print("🎤 Listening... (Speak naturally, it will wait for you to finish)")
    
    audio_buffer = []
    is_speaking = False
    silence_counter = 0
    
    # We use a blocksize of 512, which is exactly 32 milliseconds of audio
    with sd.InputStream(device=device_id, samplerate=samplerate, channels=1, callback=callback, blocksize=512):
        while True:
            # Check for any new responses from the LLM Brain
            professor_reply = brain.get_latest_response()
            if professor_reply:
                print(f"\n👨‍🏫 Professor: {professor_reply}\n")
                engine = pyttsx3.init()                               
                engine.setProperty('rate', 175) 
                engine.say(professor_reply)
                engine.runAndWait()

                del engine
                
            data_chunk = audio_queue.get()
            
            # Convert the raw numpy audio into a format the VAD model understands (Torch Tensor)
            tensor_chunk = torch.from_numpy(data_chunk.flatten()).float()
            
            # Get the probability that this 32ms chunk contains human speech (0.0 to 1.0)
            speech_prob = vad_model(tensor_chunk, samplerate).item()
            
            # If the probability is higher than 50%, the user is speaking!
            if speech_prob > 0.5:
                is_speaking = True
                silence_counter = 0 # Reset silence because they are still talking
                audio_buffer.append(data_chunk)
                
            # If they were speaking, but suddenly stopped (speech_prob < 0.5)
            elif is_speaking:
                silence_counter += 1
                audio_buffer.append(data_chunk) # Keep recording the quiet parts between words
                
                # If they have been silent for roughly 1 second (30 chunks of 32ms)
                if silence_counter > 30:
                    # They finished their sentence! Let's process it.
                    current_audio = np.concatenate(audio_buffer, axis=0)
                    wav.write("temp_chunk.wav", samplerate, current_audio)
                    
                    print("⏳ Transcribing...")
                    result = whisper_model.transcribe(
                        "temp_chunk.wav", 
                        fp16=True, 
                        initial_prompt="This is a Computer Science engineering viva. Topics include AI, Generative AI, Python, and DBMS."
                    )
                    
                    text = result["text"].strip()
                    if text:
                        print(f"🗣️ Student: {text}")
                        # Send text to the brain to process in the background
                        brain.process_transcript_async(text)
                    
                    # Reset the buffer for the next sentence
                    audio_buffer = []
                    is_speaking = False
                    silence_counter = 0

if __name__ == "__main__":
    try:
        start_listening()
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")