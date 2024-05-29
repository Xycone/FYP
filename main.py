from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import whisper
from enum import Enum
from typing import List
from tempfile import NamedTemporaryFile


torch.cuda.is_available()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = whisper.load_model("base", device=DEVICE)

class ModelSize(str, Enum):
    tiny = "tiny"
    base = "base"
    small = "small"
    medium = "medium"
    large = "large"


app = FastAPI()

@app.post("/transcribe")
async def transcribe(files: List[UploadFile] = File( ... )):
    if not files:
        raise HTTPException(status_code=400, detail="No Files Uploaded")
    
    transcripts = []

    for file in files:
        with NamedTemporaryFile(delete=True) as temp:
            with open(temp.name, "wb") as temp_file:
                temp_file.write(file.file.read())

            transcript = model.transcribe(temp.name)
            transcripts.append(
                {
                    "filename": file.filename,
                    "language": transcript["language"],
                    "transcript": transcript["text"]
                }
            )
    
    return JSONResponse(content={"transcripts": transcripts})

@app.post("/set-model-size/{size}")
async def set_model_size(size: ModelSize):
    global model
    torch.cuda.empty_cache()
    model = whisper.load_model(size, device=DEVICE)

    return JSONResponse(content={"message": f"Model size set to {size}"})