import subprocess

import torch
import numpy as np

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment

from pydub import AudioSegment

import wave
import contextlib

from sklearn.cluster import AgglomerativeClustering

class DiarisationManager:
    # Initialiser
    def __init__(self):
        self.__embedding_model = PretrainedSpeakerEmbedding(
            "speechbrain/spkrec-ecapa-voxceleb",
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    # Methods
    def is_stereo(self, file_path):
        try:
            audio = AudioSegment.from_file(file_path)
            channels = audio.channels
            return channels != 1
        except Exception as e:  
            print(f"Error: {e}")
            return False
    
    def diarise(self, file_path, segments, num_speakers):
        try:
            if file_path[-3:] != 'wav':
                file_path = self.__convert_audio(file_path)
        except Exception as e:
            print(f"Error converting audio: {e}")
            return False       
        
        # Generate speaker embeddings for each segment in segments
        embeddings = np.zeros(shape=(len(segments), 192))
        for i, segment in enumerate(segments):
            embeddings[i] = self.__segment_embedding(file_path, self.__calc_audio_duration(file_path), segment)

        embeddings = np.nan_to_num(embeddings)

        # Assign speaker to each of the segments
        for i in range(len(segments)):
            segments[i]["speaker"] = 'Speaker ' + str(self.__cluster_segments(num_speakers, embeddings)[i] + 1)

        return segments
    
    # Internal methods (can be accessed outside of class but not recommended)
    def __convert_audio(self, file_path):
        new_file_path = "converted.wav"
        subprocess.run(["ffmpeg", "-i", file_path, "converted.wav", "-y"], check=True)
        return new_file_path

    def __calc_audio_duration(self, file_path):
        with contextlib.closing(wave.open(file_path,'r')) as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / float(rate)

        return duration
    
    def __segment_embedding(self, file_path, duration, segment):
        audio = Audio()
        start = segment["start"]
        end = min(duration, segment["end"])
        clip = Segment(start, end)
        waveform, sample_rate = audio.crop(file_path, clip)

        return self.__embedding_model(waveform[None])

    def __cluster_segments(self, num_speakers, embeddings):
        clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
        labels = clustering.labels_

        return labels
