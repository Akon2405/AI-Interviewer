import os
import queue
import threading
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

# Load environment variables (e.g., GROQ_API_KEY)
load_dotenv()

class InterviewerBrain:
    def __init__(self, groq_api_key=None):
        # Prefer provided key, fallback to environment variable
        api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is missing. Please set it in your .env file or pass it directly."
            )
            
        # Initialize the Llama 3 model via Groq API
        self.llm = ChatGroq(
            api_key=api_key,
            model="llama-3.1-8b-instant",
            temperature=0.5, # Strict, analytical behavior. Lower temperature = less hallucination.
            max_tokens=150,  # Keep responses highly concise
        )
        
        # System prompt defining the strict professor persona
        self.system_prompt = (
            "You are Dr. Aris, a strict, highly experienced Computer Science Professor conducting a final-year Technical Viva. "
            "Your objective is to rigorously test the student's conceptual understanding, problem-solving skills, and fundamental knowledge.\n\n"
            "The student's input is from an automated Speech-to-Text system. You MUST ignore phonetic, grammatical, and spelling errors, "
            "focusing EXCLUSIVELY on semantic meaning and technical accuracy.\n\n"
            "CRITICAL RULES:\n"
            "1. ONE QUESTION ONLY: Ask exactly one concise, highly technical follow-up question at a time. Never give a list.\n"
            "2. EXTREME BREVITY: Keep your responses under 2 to 3 sentences maximum. Your text is processed by a live Text-to-Speech engine.\n"
            "3. NO SPOON-FEEDING: If an answer is wrong or partial, do not just give the correct answer. Ask a probing follow-up or challenge their logic.\n"
            "4. PROBE SHALLOW ANSWERS: If they recite a basic textbook definition, immediately ask how it works 'under the hood' or for a failing edge case.\n"
            "5. TONE: Professional, authoritative, and slightly intimidating. Zero friendly pleasantries, no emojis, and never break character.\n"
            "6. ZERO TOLERANCE FOR STALLING: If the student uses filler words or dodges the question, aggressively pivot and demand a direct answer.\n\n"
            "Begin by asking the student to introduce themselves and briefly explain their core project."
            )
        
        # Create the prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{user_input}")
        ])
        
        # Connect Prompt with LLM
        self.chain = self.prompt | self.llm
        
        # Store for session history to maintain conversation state
        self.store = {}
        
        # Wrap the chain with history management
        self.brain_chain = RunnableWithMessageHistory(
            self.chain,
            self.get_session_history,
            input_messages_key="user_input",
            history_messages_key="history",
        )
        
        # Concurrency setup for asynchronous "thinking" thread
        # This allows ears_v3.py to keep listening while the LLM generates a response
        self.transcript_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = None

    def get_session_history(self, session_id: str) -> InMemoryChatMessageHistory:
        """Retrieve or create a chat history for a session."""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def _thinking_worker(self, session_id):
        """Worker thread that continuously processes transcripts from the queue."""
        while not self._stop_event.is_set():
            try:
                # Wait for a transcript to be processed (1-second timeout loop)
                transcript = self.transcript_queue.get(timeout=1.0)
                
                print(f"🧠 Brain is processing: '{transcript[:30]}...'")
                
                # Hit Groq API asynchronously by offloading to this thread
                response = self.brain_chain.invoke(
                    {"user_input": transcript},
                    config={"configurable": {"session_id": session_id}}
                )
                
                # Push the professor's response to the output queue
                self.response_queue.put(response.content)
                self.transcript_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.response_queue.put(f"[Error in Brain]: {str(e)}")

    def start_thinking_thread(self, session_id="viva_session_1"):
        """Starts the background thread to process transcripts without blocking."""
        if self._worker_thread is None or not self._worker_thread.is_alive():
            self._stop_event.clear()
            self._worker_thread = threading.Thread(
                target=self._thinking_worker, 
                args=(session_id,),
                daemon=True # Daemon thread will close when the main program exits
            )
            self._worker_thread.start()
            print("🧠 Interviewer persona background thread initialized.")

    def stop_thinking_thread(self):
        """Signals the background thread to gracefully stop."""
        self._stop_event.set()
        if self._worker_thread:
            self._worker_thread.join()
            print("🧠 Interviewer persona thread stopped.")

    def process_transcript_async(self, transcript: str):
        """Feed a transcript to the brain asynchronously."""
        if transcript and transcript.strip():
            self.transcript_queue.put(transcript.strip())

    def get_latest_response(self):
        """Retrieve the latest response from the professor (non-blocking)."""
        try:
            return self.response_queue.get_nowait()
        except queue.Empty:
            return None

    def evaluate_response_sync(self, student_input: str, session_id: str = "viva_session_1") -> str:
        """Process transcript synchronously (blocking call for testing)."""
        response = self.brain_chain.invoke(
            {"user_input": student_input},
            config={"configurable": {"session_id": session_id}}
        )
        return response.content

if __name__ == "__main__":
    # Test block for running this standalone
    brain = InterviewerBrain()
    print("🧠 Brain is ready. Type your transcript (or 'quit' to exit).")
    while True:
        text = input("🗣️ Student: ")
        if text.lower() in ["quit", "exit"]:
            break
        reply = brain.evaluate_response_sync(text)
        print(f"👨‍🏫 Professor: {reply}\n")
