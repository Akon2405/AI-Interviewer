import streamlit as st
import cv2
import threading
import queue
import time
import torch
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
import whisper
import pyttsx3
from brain import InterviewerBrain
from vision_tets2 import VisionProctor 
from audio_visualizer import AudioVisualizer


st.markdown(AudioVisualizer.get_css(), unsafe_allow_html=True)

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Viva Simulator", layout="wide", initial_sidebar_state="collapsed")

# --- GLOBAL UI OVERRIDES ---
# --- GLOBAL UI OVERRIDES ---
custom_css = """
<style>
    /* 1. Remove the massive empty space at the top of the page */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 0rem !important;
        max-width: 95% !important;
    }
    
    /* 2. AGGRESSIVELY hide the main browser scrollbar */
    html, body, [data-testid="stAppViewContainer"] {
        overflow-y: hidden !important; 
    }
    
    /* 3. Refine the Buttons */
    .stButton > button {
        border-radius: 25px !important;
        font-weight: 600 !important;
        border: 1px solid #4facfe !important;
        transition: all 0.3s ease-in-out !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 15px rgba(79, 172, 254, 0.4) !important;
        transform: scale(1.02);
    }
    
    /* 4. Force rounded corners on Streamlit Containers (like the Chat Box) */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 20px !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# --- INITIALIZE CORES, QUEUES & THREAD EVENTS ---
# We store a threading.Event object in session state so it persists across UI reruns
if "prof_speaking" not in st.session_state:
    st.session_state.prof_speaking = threading.Event()
if "last_speaking_state" not in st.session_state:
    st.session_state.last_speaking_state = False
if "stop_event" not in st.session_state:
    st.session_state.stop_event = threading.Event()
if "text_queue" not in st.session_state:
    st.session_state.text_queue = queue.Queue()
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to your Viva. I am ready when you are. Please introduce yourself."}
    ]
if "listening_active" not in st.session_state:
    st.session_state.listening_active = False

# --- AUDIO & PIPELINE INITIALIZATION (Cached for performance) ---
@st.cache_resource
def load_pipeline_models():
    device_id = 4 
    samplerate = 16000
    
    st.write("")
    whisper_model = whisper.load_model("small")
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', trust_repo=True)
    brain = InterviewerBrain()
    brain.start_thinking_thread()
    
    return whisper_model, vad_model, brain, device_id, samplerate

whisper_model, vad_model, brain, device_id, samplerate = load_pipeline_models()

# --- THE BACKGROUND AUDIO ENGINE ---
# FIX: Pass the stop_event directly as a parameter so the thread checks a local memory flag
def audio_listening_worker(text_q, stop_event):
    audio_queue = queue.Queue()
    
    def callback(indata, frames, time, status):
        audio_queue.put(indata.copy())
        
    audio_buffer = []
    is_speaking = False
    silence_counter = 0
    
    with sd.InputStream(device=device_id, samplerate=samplerate, channels=1, callback=callback, blocksize=512):
        # FIX: The thread runs continuously until the stop event is explicitly set
        while not stop_event.is_set():
            try:
                data_chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue
                
            tensor_chunk = torch.from_numpy(data_chunk.flatten()).float()
            speech_prob = vad_model(tensor_chunk, samplerate).item()
            
            if speech_prob > 0.5:
                is_speaking = True
                silence_counter = 0
                audio_buffer.append(data_chunk)
            elif is_speaking:
                silence_counter += 1
                audio_buffer.append(data_chunk)
                
                if silence_counter > 30:
                    current_audio = np.concatenate(audio_buffer, axis=0)
                    wav.write("temp_web_chunk.wav", samplerate, current_audio)
                    
                    result = whisper_model.transcribe("temp_web_chunk.wav", fp16=True)
                    text = result["text"].strip()
                    
                    if text:
                        text_q.put({"sender": "student", "text": text})
                        brain.process_transcript_async(text)
                        
                        while not stop_event.is_set():
                            reply = brain.get_latest_response()
                            if reply:
                                text_q.put({"sender": "professor", "text": reply})
                                break
                            time.sleep(0.1)
                            
                    audio_buffer = []
                    is_speaking = False
                    silence_counter = 0


def speak_in_background(text,speaking_event):
    """Spins up a temporary background thread to speak text without freezing the camera"""
    def _speak():
        speaking_event.set()


        engine = pyttsx3.init()
        engine.setProperty('rate', 175)
        engine.say(text)
        engine.runAndWait()
        del engine

        speaking_event.clear()
        
    threading.Thread(target=_speak, daemon=True).start()

# --- MAIN INTERFACE SPLIT ---
left_col, right_col = st.columns([7, 3])

# ==========================================
# LEFT COLUMN: MAIN AUDIO INTERFACE (BIG BOX)
# ==========================================
with left_col:
    st.title("AI Interviewer")
    #st.markdown("---")
    
    st.subheader("Professor --")
    
    visualizer_container = st.container(border=False, height=450)
    with visualizer_container:
        st.write("")
        st.write("")
        wave_placeholder = st.empty()
        
        if not st.session_state.listening_active:
            # Replaced the red error box with the idle circular visualizer!
            wave_placeholder.markdown(AudioVisualizer.get_wave_html(False), unsafe_allow_html=True)


    # --- REDESIGNED CONTROL PANEL ---
    st.markdown("#### Session Controls")
    control_col1, control_col2, control_col3 = st.columns([3, 3, 4]) # 3 columns for better spacing
    
    with control_col1:
        if not st.session_state.listening_active:
            if st.button("🎙️ Start Session", use_container_width=True, type="primary"):
                st.session_state.listening_active = True
                st.session_state.stop_event.clear()
                
                t = threading.Thread(
                    target=audio_listening_worker, 
                    args=(st.session_state.text_queue, st.session_state.stop_event), 
                    daemon=True
                )
                t.start()
                st.rerun()
        else:
            if st.button("🛑 Terminate", use_container_width=True):
                st.session_state.listening_active = False
                st.session_state.stop_event.set()
                if "proctor" in st.session_state:
                    st.session_state.proctor.stop_camera()
                    del st.session_state.proctor
                st.rerun()
                
    with control_col2:
        # A sleek, compact status pill instead of a massive bar
        if st.session_state.listening_active:
            active_pill = """
            <div style='display: flex; align-items: center; justify-content: center; height: 100%; padding-top: 5px;'>
                <div style='background: rgba(46, 204, 113, 0.2); color: #2ecc71; border: 1px solid #2ecc71; padding: 6px 15px; border-radius: 20px; font-weight: bold; font-size: 14px; box-shadow: 0 0 8px rgba(46, 204, 113, 0.4);'>
                    🎙️ Listening...
                </div>
            </div>
            """
            st.markdown(active_pill, unsafe_allow_html=True)
        else:
            standby_pill = """
            <div style='display: flex; align-items: center; justify-content: center; height: 100%; padding-top: 5px;'>
                <div style='background: rgba(128, 128, 128, 0.2); color: #888; border: 1px solid #888; padding: 6px 15px; border-radius: 20px; font-weight: bold; font-size: 14px;'>
                    💤 Mic Standby
                </div>
            </div>
            """
            st.markdown(standby_pill, unsafe_allow_html=True)

# ==========================================
# RIGHT COLUMN: SIDEBAR FEEDS (ZOOM STYLE)
# ==========================================
with right_col:
    st.subheader(" Live Proctoring")
    
    camera_placeholder = st.empty()
    # Kept so the active loop doesn't crash, but left blank when idle!
    
    if not st.session_state.listening_active:
        # Changed aspect-ratio to 16/9 to make it shorter and wider!
        # Added border-radius: 15px to match the new rounded UI
        offline_camera_html = """
        <div style="background-color: #1a1c23; border-radius: 15px; width: 100%; aspect-ratio: 16/9; display: flex; align-items: center; justify-content: center; border: 1px solid #333; box-shadow: inset 0 0 20px rgba(0,0,0,0.5);">
            <svg viewBox="0 0 24 24" fill="#444" style="width: 25%; opacity: 0.8;">
                <path d="M12 12c2.21 0 4-1.79 4-4s-1.79-4-4-4-4 1.79-4 4 1.79 4 4 4zm0 2c-2.67 0-8 1.34-8 4v2h16v-2c0-2.66-5.33-4-8-4z"/>
            </svg>
        </div>
        """
        camera_placeholder.markdown(offline_camera_html, unsafe_allow_html=True)
        # Note: The idle pill and the st.markdown("---") line have been completely deleted!
    
    # We add a tiny bit of invisible space just to separate the camera and chat
    st.write("") 
    
    st.subheader("Examination Logs")
    
    # Increased height from 300 to 420 since the camera is smaller and the idle button is gone
    chat_container = st.container(height=400, border=True) 
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

# --- DYNAMIC INTERFACE REFRESHER ---
if st.session_state.listening_active:
    if "proctor" not in st.session_state:
        st.session_state.proctor = VisionProctor()
        st.session_state.proctor.start_camera()
        
    try:
        # Loop endlessly while the session is active
        while not st.session_state.stop_event.is_set():

            current_speaking_state = st.session_state.prof_speaking.is_set()
            
            # Only redraw the HTML if the professor just started or just stopped speaking
            # (If we redraw it every frame, the animation will flicker and reset)
            if current_speaking_state != st.session_state.last_speaking_state:
                wave_html = AudioVisualizer.get_wave_html(current_speaking_state)
                wave_placeholder.markdown(wave_html, unsafe_allow_html=True)
                st.session_state.last_speaking_state = current_speaking_state
            
            # 1. PROCESS VIDEO AND EYE TRACKING
            frame, status = st.session_state.proctor.process_next_frame()
            
            if frame is not None:
                # 1. Crop to perfect 16:9 aspect ratio (Your existing code)
                h, w, _ = frame.shape
                target_ratio = 16 / 9
                current_ratio = w / h
                
                if current_ratio > target_ratio: 
                    new_w = int(h * target_ratio)
                    offset = (w - new_w) // 2
                    frame = frame[:, offset:offset+new_w]
                elif current_ratio < target_ratio:
                    new_h = int(w / target_ratio)
                    offset = (h - new_h) // 2
                    frame = frame[offset:offset+new_h, :]

                # --- NEW: DRAW THE HUD OVERLAY DIRECTLY ON THE VIDEO ---
                # Recalculate dimensions after cropping
                h, w, _ = frame.shape 
                
                # Create a semi-transparent black bar at the bottom
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, h - 40), (w, h), (0, 0, 0), -1)
                alpha = 0.5 # Transparency level (0.0 to 1.0)
                cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

                # Determine color and text based on status (Note: frame is in RGB format here!)
                if status == "CENTER":
                    text_color = (0, 255, 0) # Green
                    display_text = " Focus Status: Center"
                else:
                    text_color = (255, 50, 50) # Red
                    display_text = f" Warning: {status}"

                # Draw the text onto the semi-transparent bar
                cv2.putText(frame, display_text, (15, h - 13), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2, cv2.LINE_AA)
                # -------------------------------------------------------

                # 2. Push the finalized, watermarked frame to the UI
                camera_placeholder.image(frame, channels="RGB", use_container_width=True)
                
                # (Notice we completely removed the proctor_status_placeholder.success logic!)

            # 2. CHECK FOR AUDIO/TEXT MESSAGES
            try:
                while not st.session_state.text_queue.empty():
                    new_data = st.session_state.text_queue.get_nowait()
                    if new_data["sender"] == "student":
                        st.session_state.messages.append({"role": "user", "content": new_data["text"]})
                    elif new_data["sender"] == "professor":
                        st.session_state.messages.append({"role": "assistant", "content": new_data["text"]})
                        
                        # Trigger the voice in the background!
                        speak_in_background(new_data["text"], st.session_state.prof_speaking)
                        
                    st.rerun() # Rerun to update chat bubbles
            except queue.Empty:
                pass
                
            # 3. PACE THE LOOP (~30 FPS for smooth video)
            time.sleep(0.03) 
            
    except Exception as e:
        st.error(f"Error in UI loop: {e}")

# Clean up hardware when session ends
else:
    if "proctor" in st.session_state:
        st.session_state.proctor.stop_camera()
        del st.session_state.proctor