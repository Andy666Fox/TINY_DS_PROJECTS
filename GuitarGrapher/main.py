from email.mime import audio
import imp
import pickle
from pyexpat import model
import numpy as np 
from abc import ABC
import librosa
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.sequence import pad_sequences
import os


class MEngine(ABC):
    def __init__(self):
        self.model = pickle.load(open('model.sav', 'rb'))
        self.label_encoder = LabelEncoder()
        
    
    def get_data_from_notes(self, path_to_files: str):
        sound_data = []
        files_list = os.listdir(path_to_files)
        X = []
        
        for file in files_list:
            temp_audio_data = list()
            audio_data, samplerate = librosa.core.load(path_to_files + file, res_type='kaiser_fast')
            spectral_data = librosa.feature.spectral_bandwith(y=audio_data, sr=samplerate)
            
            for sd in spectral_data:
                temp_audio_data.append(sd)
                
            sound_data.append(temp_audio_data)
            
        for s in sound_data:
            for val in s:
                X.append(val)
                
        max_len = max([len(X) for x in X])
        
        X = pad_sequences(X, maxlen=max_len)
        X = X.reshape(1, -1)
        
        return X
        
        
        
        
        
    def get_predicted_notes(self, to_predict: np.array) -> list:
        return model.predict(to_predict)
    