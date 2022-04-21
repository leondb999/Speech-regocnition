import os
import pathlib
import shutil

import h5py

import numpy as np

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

#sys.path.append(r"")

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
    audio, _  = tf.audio.decode_wav(contents=audio_binary)
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

#TODO Convert .wavFile to Mono
origins = ["*"]

file_path_mono = r"C:/Users/Alessandro Avanzato/OneDrive/Desktop/SpeaQ/audio.wav"

#file_path_mono = r"C:/Users/Alessandro Avanzato/OneDrive/Desktop/SpeaQ/00176480_nohash_0.wav"

#file_path = r"C:/Users/led51/PycharmProjects/pythonProject5/ale_cat.wav"
#file_path = r"C:/Users/led51/PycharmProjects/pythonProject5/cat_mono.wav"
#file_path = r"C:/Users/led51/PycharmProjects/pythonProject5/00176480_nohash_0.wav"
input_shape = (129, 1)



#sound = AudioSegment.from_file_using_temporary_files(file_path)
#sound = sound.set_channels(1)
#file_path_mono = sound.export(r"C:/Users/Alessandro Avanzato/OneDrive/Desktop/Test/test.wav", format="wav")

#stereo_audio = AudioSegment.from_file(file_path_mono, format="wav")
#mono_audios = stereo_audio.split_to_mono()
#mono_left = stereo_audio[0].export(file_path_mono)
#commands = np.array(['cat' 'bed' 'bird' 'house' 'dog'])
commands = ['cat', 'bed', 'bird', 'house', 'dog']
AUTOTUNE = tf.data.AUTOTUNE

num_labels = len(commands)
print("num_labels: ", num_labels)
files_ds_list = tf.random.shuffle([str(file_path_mono)])

files_ds = tf.data.Dataset.from_tensor_slices(files_ds_list)
print(np.array(files_ds))
label="cat"

waveform_ds = files_ds.map(
    map_func=get_waveform_and_label,
    num_parallel_calls=AUTOTUNE)

spectrogram_ds = waveform_ds.map(
    map_func=get_spectrogram_and_label_id,
    num_parallel_calls=AUTOTUNE)



model = tf.keras.models.load_model("model1.h5")

print("my_saved_model.h5: ", model)


#file_path = "/content/data/backward/017c4098_nohash_0.wav"

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

def ergebnis_auswerten(result):
    sigmoid_list = []
    # Bringe alle prediction Werte auf eine Scala

    for prediction in result:
        calc = sigmoid(prediction)
        sigmoid_list.append(calc)
    print("sigmoid_list: ", sigmoid_list)

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
    # if(value < 0.1):
    #   print("word konnte nicht erkannt werden, spreche das richtige label")
    # else:
    print("value: ", value, "label:", commands[index - 1], ", index: ", index - 1)
    return value



prediction_ergebnis = ergebnis_berechnen(file_path_mono)
normalisierte_prediction =ergebnis_auswerten(prediction_ergebnis)
print("Hello: ", normalisierte_prediction )