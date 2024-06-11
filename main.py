from fastapi import FastAPI, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse

import torch

from typing import List
from model.transcriptionManager import TranscriptionManager
from model.diarisationManager import DiarisationManager
from dto.transcriptionDTO import TranscriptionDTO
from tempfile import NamedTemporaryFile

# Create model_manager object and load the default "base" model
whisper = (
    TranscriptionManager(
        "base",
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    .load_model()
)

diariser = DiarisationManager()

app = FastAPI()

# Embedding model used for speaker diarisation requires audio file to be mono (single channel)
@app.post("/transcribe-files")
async def transcribe(form_data: TranscriptionDTO = Depends(), files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No Files Uploaded")
    
    # Replaces the currently loaded model with the new model of the specified size
    if form_data.model_size != whisper.get_size():
        whisper.unload_model().set_size(form_data.model_size).load_model()

    response = []

    for file in files:
        with NamedTemporaryFile(delete=True) as temp:
            # Copies the uploaded audio file to the temporary file
            with open(temp.name, 'wb') as temp_file:
                temp_file.write(file.file.read())
            
            if form_data.diarisation and diariser.is_stereo(temp.name):
                raise HTTPException(status_code=400, detail="Audio input must be mono (single channel)")
            
            results = whisper.transcribe(temp.name)

            if form_data.diarisation:                      
                segments = diariser.diarise(temp.name, results["segments"], form_data.num_speakers)
            else:
                segments = results["segments"]

            response.append(
                {
                    "filename": file.filename,
                    "language": results["language"],
                    "segments": segments
                }
            )
            
    return JSONResponse(content={"transcripts": response})
