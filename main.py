
from re import X


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

#model_dir = 'C:/2019-Leon-eigene-Dateien/Studium/6 Semester/Integrationsseminar/Integration/DHBW/model.h5'
model = tf.keras.models.load_model(
    'model.h5',
    custom_objects=None, compile=True)



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

class Anfrage(BaseModel):
    filename: str
    bytes : bytes

class PydanticFile(BaseModel):
    file: UploadFile = File(...)

def current_milli_time():
    return str(round(time.time() * 1000))

@api_router.post("/anfrage/")
#async def create_anfrage(file: UploadFile=File(...)):
async def create_anfrage(file: UploadFile = File(...)):
    path= r"C:/2019-Leon-eigene-Dateien/Studium/6-Semester/Integrationsseminar/Speech-regocnition/audio_files/"+current_milli_time() + "audio.wav"
    with open(path, 'wb') as audio_file:
        content = await file.read()
        audio_file.write(content)
        audio_file.close()
        
    print("audio path: ", path)
    return {"filename": audio_file}

app.include_router(api_router)