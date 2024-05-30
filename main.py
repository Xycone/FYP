from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse

import torch

from typing import List
from model.whisperManager import WhisperManager
from dto.transcriptionDTO import TranscriptionDTO
from tempfile import NamedTemporaryFile

# Create model_manager object and load the default "base" model
whisper = (
    WhisperManager(
        "base",
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    .load_model()
)

app = FastAPI()

@app.post("/transcribe")
async def transcribe(form_data: TranscriptionDTO = Depends(), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No Files Uploaded")
    
    if form_data.model_size != whisper.get_size():
        # Update model size and load the new models
        whisper.unload_model().set_size(form_data.model_size).load_model()

    response = []

    for file in files:
        with NamedTemporaryFile(delete=True) as temp:
            with open(temp.name, 'wb') as temp_file:
                temp_file.write(file.file.read())
            
            transcript = whisper.transcribe(temp.name)
            response.append(
                {
                    "filename": file.filename,
                    "language": transcript["language"],
                    "transcript": transcript["text"]
                }
            )
    
    return JSONResponse(content={"transcripts": response})
    