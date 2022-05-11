
from re import X

import os
import pathlib
import shutil

import h5py

import numpy as np
import librosa
import seaborn as sns
import tensorflow as tf
import wave
import math
import ffmpeg
import sys
import requests
import json
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display
from pydub import AudioSegment
#import  tensorflow as tf
from pydantic import BaseModel
from fastapi import FastAPI,Request,Form,File, APIRouter, UploadFile
from numpy import argmax
from numpy import max
from numpy import array
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import tensorflow as tf
import keras
#print(tf.version.VERSION)
import io
import time
from pydub import AudioSegment


def get_label(file_path):
    print("file_path: ", file_path)
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]

def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to float32 tensors, normalized
    # to the [-1.0, 1.0] range. Return float32 audio and a sample rate.
    #audio, _  = tf.audio.decode_wav(contents=audio_binary, desired_channels=1)
    audio, _  = tf.audio.decode_wav(contents=audio_binary, desired_channels= -1, desired_samples = -1)
    # Since all the data is single channel (mono), drop the channels
    # axis from the array.
    return tf.squeeze(audio, axis=-1)

def get_waveform_and_label(file_path):
   # label = get_label(file_path)
    print("label",label)
    audio_binary = tf.io.read_file(file_path)
    print("audio_binary", audio_binary)
    waveform = decode_audio(audio_binary)
    print("waveform", waveform)
    return waveform, label

def get_spectrogram(waveform):
    # Zero-padding for an audio waveform with less than 16,000 samples.
    input_len = 16000
    waveform = waveform[:input_len]
    zero_padding = tf.zeros(
        [16000] - tf.shape(waveform),
        dtype=tf.float32)
    # Cast the waveform tensors' dtype to float32.
    waveform = tf.cast(waveform, dtype=tf.float32)
    # Concatenate the waveform with zero_padding, which ensures all audio
    # clips are of the same length.
    equal_length = tf.concat([waveform, zero_padding], 0)
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(
        equal_length, frame_length=255, frame_step=128)
    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a channels dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (batch_size, height, width, channels).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id

def get_waveform_and_label(file_path):
   # label = get_label(file_path)
    print("label",label)
    audio_binary = tf.io.read_file(file_path)
    print("audio_binary", audio_binary)
    waveform = decode_audio(audio_binary)
    print("waveform", waveform)
    return waveform, label

def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id

def preprocess_dataset(files):
    files_ds = tf.data.Dataset.from_tensor_slices(files)
    output_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE)
    print("output_ds: ", output_ds)
    output_ds = output_ds.map(
        map_func=get_spectrogram_and_label_id,
        num_parallel_calls=AUTOTUNE)
    return output_ds


def ergebnis_berechnen(wavdatei):
    #########

    #Hier muss binäre Datei wieder in wav umgewandelt werden

    ##########
    # 1. Deklariere Liste für  Audio Datein
    audio_file_list = []

    # 2. Füge Dateipfad als String zur Liste hinzu # go
    audio_file_list.append(str(wavdatei))

    # 3. Konvertiere Liste zu EagerTensor
    audio_file_list = tf.random.shuffle(audio_file_list)

    wavefile = audio_file_list[0]

    # 4. Übergebe Liste von EagerTensoren an Preprocessing Funktion & speichere Ergebnis in einer neuen liste
    audio_ds = preprocess_dataset(audio_file_list) # Preprocessen Liste von EagerTensoren
    print("audio_ds: ", audio_ds)


    # 5. Füge Preprocesste Audiodateien als num
    list_audio = [] # type sample_ds:  <class 'numpy.ndarray'>

    for audio, label in audio_ds:

        list_audio.append(audio.numpy())

    # 6. Wandle liste in NumpyArray um
    list_audio = np.array(list_audio) #list_audio <class 'numpy.ndarray'>
   # print("list_audio: ",list_audio)
    #y_pred = np.argmax(model.predict(test_audio), axis=1)

    y_pred = np.argmax(model.predict(list_audio), axis=1)

    # Todo ergebnis

   # resut = model.predict(list_audio)[0]
    #result =  model.predict(list_audio)[0]
    return model.predict(list_audio)[0]
# Sigmpoid Funktion


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def ergebnis_auswerten(result, label_index):
    sigmoid_list = []
    # Bringe alle prediction Werte auf eine Scala
    # Konvertiere Prediction in Prozentzahl
    for prediction in result:
        calc = sigmoid(prediction)
        sigmoid_list.append(calc)
    print("sigmoid_list: ", sigmoid_list)
    value_pred = sigmoid_list[label_index]
    label  = commands[label_index]

    '''
    # Display highest pred value & index to get label from 'commands' list
    index = 0
    value = 0
    zw_index = 0
    for prediction in sigmoid_list:
        zw_index +=1
        print("prediction", prediction)
        if value < prediction:
            value = prediction
            index = zw_index
            print("index: ", index)
    print("commands: ", commands)
    print("value: ", value)
    '''
    # if(value < 0.1):
    #   print("word konnte nicht erkannt werden, spreche das richtige label")
    # else:
    #label = commands[index - 1]

    print("value_pred: ", value_pred, "label:", label)
    return [value_pred, label]




#model_dir = 'C:/2019-Leon-eigene-Dateien/Studium/6 Semester/Integrationsseminar/Integration/DHBW/model.h5'asddas

app = FastAPI()
api_router = APIRouter()
origins = [
    "http://localhost",
    "http://localhost:8000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



"""
@app.get("/")
async def main():
    return {"Hello": "World"}
            @app.get("/")
            def main(request: Request):
                client_host = request.client.host
                data=request.
                return {"Hello": request.titel}
                blob: bytes=File(...)
"""


class PydanticFile(BaseModel):
    file: UploadFile = File(...)

def current_milli_time():
    return str(round(time.time() * 1000))

model = tf.keras.models.load_model('model.h5',custom_objects=None, compile=True)
origins = ["*"]
input_shape = (129, 1)
commands = ['cat', 'bed', 'bird', 'house', 'dog']
AUTOTUNE = tf.data.AUTOTUNE
num_labels = len(commands)
print("num_labels: ", num_labels)
label="cat"


def read_wav(path, sr, duration=None, mono=True):
    wav, _ = librosa.load(path) #librosa.load(path, mono=mono, sr=sr, duration=duration)
    return wav

class Anfrage(BaseModel):
    file: UploadFile = File(...)
    label_index: str

@api_router.post("/anfrage/")
#async def create_anfrage(file: UploadFile=File(...)):
#async def create_anfrage(file: UploadFile = File(...), anfrage: Anfrage):
async def create_anfrage(file: UploadFile = File(...)):
    path= r"C:\Users\Alessandro Avanzato\github\Speech-regocnition\audio_files" + current_milli_time() + "audio.wav"

    label_index = int(file.filename)
    print("----------------------label_index:", label_index)
    #Erstelle Wav File
    with open(path, 'wb') as audio_file:
        content = await file.read()
        print("type:", type(content))
        audio_file.write(content)
        audio_file.close()

    #Konvertiere Wav Datei zu Mono indem channels zu 1 gesetzt werden
    sound = AudioSegment.from_wav(path)
    sound = sound.set_channels(1)
    sound.export(path, format="wav")

    print("sound channels:", sound.channels)
    files_ds_list = tf.random.shuffle([str(path)])
    files_ds = tf.data.Dataset.from_tensor_slices(files_ds_list)

    print(np.array(files_ds))
    label="cat"

    waveform_ds = files_ds.map(
        map_func=get_waveform_and_label,
        num_parallel_calls=AUTOTUNE)

    spectrogram_ds = waveform_ds.map(
        map_func=get_spectrogram_and_label_id,
        num_parallel_calls=AUTOTUNE)

    print("my_saved_model.h5: ", model)
    prediction_ergebnis = ergebnis_berechnen(path)
    result_list = ergebnis_auswerten(prediction_ergebnis,label_index)
    print("commands: ", commands)
    print("Hello: ", result_list)

    return {"filename": result_list[0], 'label': result_list[1]}

    #return {"filename": "hi"}
app.include_router(api_router)
