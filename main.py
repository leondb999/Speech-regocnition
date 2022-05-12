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
from pydantic import BaseModel
from fastapi import FastAPI,Request,Form,File, APIRouter, UploadFile
from numpy import argmax
from numpy import max
from numpy import array
from fastapi.middleware.cors import CORSMiddleware
import aiofiles
import tensorflow as tf
import keras
import io
import time
from pydub import AudioSegment



# Definieren Sie eine Funktion, die Etiketten unter Verwendung der übergeordneten Verzeichnisse für jede Datei erstellt:
# - Teilen Sie die Dateipfade in tf.RaggedTensors auf (Tensoren mit unregelmäßigen Abmessungen – mit Abschnitten, die unterschiedliche Längen haben können).
def get_label(file_path):
    print("file_path: ", file_path)
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    # Note: You'll use indexing here instead of tuple unpacking to enable this
    # to work in a TensorFlow graph.
    return parts[-2]

#Lassen Sie uns nun eine Funktion definieren, die die rohen WAV-Audiodateien des Datensatzes in Audiotensoren vorverarbeitet:
def decode_audio(audio_binary):
    # Decode WAV-encoded audio files to float32 tensors, normalized
    # to the [-1.0, 1.0] range. Return float32 audio and a sample rate.
    #audio, _  = tf.audio.decode_wav(contents=audio_binary, desired_channels=1)
    audio, _  = tf.audio.decode_wav(contents=audio_binary, desired_channels= -1, desired_samples = -1)
    # Since all the data is single channel (mono), drop the channels
    # axis from the array.
    return tf.squeeze(audio, axis=-1)

#Definieren Sie eine weitere Hilfsfunktion get_waveform_and_label– – die alles zusammenfasst:
# - Die Eingabe ist der WAV-Audiodateiname.
# - Die Ausgabe ist ein Tupel, das die Audio- und Label-Tensoren enthält, die für überwachtes Lernen bereit sind.
def get_waveform_and_label(file_path):
  
    audio_binary = tf.io.read_file(file_path)
    print("audio_binary", audio_binary)
    waveform = decode_audio(audio_binary)
    print("waveform", waveform)
    return waveform, label

# Konvertiere Wellenformen (Wav) in Spektorgramme
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

#Definieren Sie nun eine Funktion, die den Wellenformdatensatz in Spektrogramme und ihre entsprechenden Beschriftungen als Integer-IDs umwandelt:
def get_spectrogram_and_label_id(audio, label):
    spectrogram = get_spectrogram(audio)
    label_id = tf.argmax(label == commands)
    return spectrogram, label_id

#Definieren Sie eine weitere Hilfsfunktion get_waveform_and_label– – die alles zusammenfasst:
# - Die Eingabe ist der WAV-Audiodateiname.
# - Die Ausgabe ist ein Tupel, das die Audio- und Label-Tensoren enthält, die für überwachtes Lernen bereit sind.
def get_waveform_and_label(file_path):
   
    audio_binary = tf.io.read_file(file_path)
    print("audio_binary", audio_binary)
    waveform = decode_audio(audio_binary)
    print("waveform", waveform)
    return waveform, label

# Preprocessing (verwende die obengenannten Funktionen)
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

# Result Calculation mit Preprocessing &  prediciton (model.predict) 
def ergebnis_berechnen(wavdatei):
    #Hier muss binäre Datei wieder in wav umgewandelt werden
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
    list_audio = [] 
    for audio, label in audio_ds:
        list_audio.append(audio.numpy())

    # 6. Wandle liste in NumpyArray um
    list_audio = np.array(list_audio) 
    y_pred = np.argmax(model.predict(list_audio), axis=1)

    return model.predict(list_audio)[0]
    
# Sigmpoid Funktion 
def sigmoid(x):
    return 1 / (1 + math.exp(-x))

#Gibt die Vorhersage für einen bestimmten Index zurück
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
    print("value_pred: ", value_pred, "label:", label)
    return [value_pred, label]

# Server von FastAPI
# Umgehung der CORS Policy
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

#Anteile Zeit in Millisekunden
def current_milli_time():
    return str(round(time.time() * 1000))

# Lade TensorFlow Modell in Programm
model = tf.keras.models.load_model('model.h5',custom_objects=None, compile=True)
origins = ["*"]
input_shape = (129, 1)
commands = ['cat', 'bed', 'bird', 'house', 'dog']
AUTOTUNE = tf.data.AUTOTUNE
num_labels = len(commands)
print("num_labels: ", num_labels)
label=""



@api_router.post("/anfrage/")
async def create_anfrage(file: UploadFile = File(...)):
    #Leon path = C:\2019-Leon-eigene-Dateien\Studium\6-Semester\Integrationsseminar\Speech-regocnition\audio_files
    #path= r"C:\Users\Alessandro Avanzato\github\Speech-regocnition\audio_files" + current_milli_time() + "audio.wav"

    #TODO Bitte eigenen Pfad zum audio_files Ordner hinzufügen
    path= r"C:/2019-Leon-eigene-Dateien/Studium/6-Semester/Integrationsseminar/Speech-regocnition/audio_files/" + current_milli_time() + "audio.wav"
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
    
    #Berechne Ergebnis
    prediction_ergebnis = ergebnis_berechnen(path)
    result_list = ergebnis_auswerten(prediction_ergebnis,label_index)
    print("Labels: ", commands)
    print("Prediction & Label: ", result_list)

    return {"filename": result_list[0], 'label': result_list[1]}

app.include_router(api_router)
