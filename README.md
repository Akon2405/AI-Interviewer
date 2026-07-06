# AI Interviewer 

An advanced, multithreaded AI interviewing application designed to conduct rigorous Technical Vivas. Built with Streamlit, this simulator features real-time voice activity detection, local Whisper transcription, and computer-vision-based eye tracking to monitor student focus. Mainly focuses on computer science projects and related branches.

## Hardware Requirements (Important)
Because this application runs heavy inference loops in real-time without relying on paid cloud APIs (aside from Groq), **you must have a dedicated GPU** to experience smooth 30-FPS rendering and instant audio transcription.
* **Recommended:** NVIDIA GPU with CUDA support.
* **CPU Only:** *Not recommended.* Running on a standard CPU will cause severe bottlenecking, leading to delayed audio transcription and frozen video frames.

## Core Architecture
This application utilizes a decoupled, multithreaded architecture to ensure a seamless frontend UI without being blocked by heavy AI inferences:
* **The Main Thread:** Manages the Streamlit UI, live OpenCV webcam rendering, and dynamic CSS animations.
* **Audio Worker Thread:** Continuously loops hardware microphone buffers, utilizing Silero VAD to detect speech and OpenAI Whisper for local transcription.
* **Brain Engine Thread:** Connects to the Groq LLM (configured as a strict CS Professor) to evaluate answers and generate technical follow-up questions.

## Features
- **Live Proctoring:** Uses Google MediaPipe Face Landmarkers to track eye-blendshapes and detect if the student is looking away from the screen.
- **Dynamic Audio Visualizer:** Pure CSS/HTML visualizers that react instantly to the text-to-speech engine.
- **Strict Examiner Persona:** Prompt-engineered to ignore phonetic transcription errors, prevent spoon-feeding, and push students on edge cases.

Installation & Setup

Follow these steps to get the environment running on your local machine.
1. Prerequisites

    Python 3.10+ installed on your system.

    NVIDIA GPU (recommended for smooth real-time performance) or Apple Silicon (M1/M2/M3).

    Git installed on your machine.

2. Clone the Repository
Bash

git clone https://github.com/Akon2405/AI-Interviewer.git
cd AI-Viva-Simulator

3. Setup Virtual Environment

It is highly recommended to use a virtual environment to avoid dependency conflicts:

    Windows:
    Bash

    python -m venv venv
    venv\Scripts\activate

    macOS/Linux:
    Bash

    python -m venv venv
    source venv/bin/activate

4. Install Dependencies

Install all required libraries via pip:
Bash

pip install -r requirements.txt

5. Configure Environment Variables

You need an API key from Groq for the "Strict Professor" brain to function.

    Create a file named .env in the root directory.

    Add your API key inside:
    Plaintext

    GROQ_API_KEY=your_actual_api_key_here

6. Download AI Model Assets

This project requires the Google MediaPipe Face Landmarker task file.

    Download face_landmarker.task from the official MediaPipe Models repository.

    Place the file directly in the root project folder (next to app.py).

7. Launch the Simulator

Run the application using Streamlit:
Bash

streamlit run app.py

Once the local server starts, open the URL displayed in your terminal (usually http://localhost:8501) in your web browser.
⚠️ Troubleshooting

    "Module Not Found": Ensure your virtual environment is active (you should see (venv) in your terminal prompt) before running pip install or streamlit run.

    Audio Lag: If the voice-to-voice interaction feels slow, ensure your PyTorch is configured to use your GPU. You can verify this by running import torch; print(torch.cuda.is_available()) in a Python shell.

    Camera Not Opening: Ensure no other application (like Zoom or Teams) is using your webcam, as only one process can access the camera hardware at a time.
