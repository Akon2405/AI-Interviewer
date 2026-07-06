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

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/AI-Viva-Simulator.git](https://github.com/YourUsername/AI-Viva-Simulator.git)
   cd AI-Viva-Simulator
