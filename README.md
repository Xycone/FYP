# Introduction
This is a backend application developed for part of my 3 months long Final Year Project. The application, developed using the FastAPI web framework, provides API endpoints for transcribing audio files using OpenAI's Whisper ASR Model.

## Installation
To install and run the Audio File Transcription Application, follow these steps:
1. Clone the repository to your local machine.
2. Install Docker
3. Build the Docker image. Replace <image_name> with the desired name for your Docker image.
Run ```docker build -t <image_name> .``` inside the Command Line Interface (CLI).
4. Run the Docker Container. Replace <image_name> with the name u gave to your Docker image.
Run ```docker run -p 8000:8000 <image_name>``` inside the CLI.
5. Access the Swagger interface by going to ```http://localhost:8000/docs``` in your web browser



