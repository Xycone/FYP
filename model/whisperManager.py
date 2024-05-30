import whisper
import torch

from model.modelSizes import ModelSizes
from model.deviceTypes import DeviceTypes

class WhisperManager:
    # Initialiser
    def __init__(self, size: ModelSizes, device: DeviceTypes):
        self.__size = size
        self.__device = torch.device(device)
        self.__model = None

    # Getters & Setters
    def get_size(self):
        return self.__size
    
    def set_size(self, size: ModelSizes):
        self.__size = size
        return self

    # Functions
    def load_model(self):
        self.__model = whisper.load_model(self.__size, self.__device)
        return self
    
    def unload_model(self):   
        if self.__device == "cuda":
            torch.cuda.empty_cache()

        self.__model = None
        return self
    
    def transcribe(self, file_path):
        if self.__model is None:    
            raise RuntimeError("Model has not been loaded. Call load_model() first.")
        
        try:
            transcript = self.__model.transcribe(file_path)
        except whisper.ModelNotFoundError as e:
            raise RuntimeError(f"Model not found: {e}")
        
        return transcript