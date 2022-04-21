
from re import X


#import  tensorflow as tf
from pydantic import BaseModel
from fastapi import FastAPI,Request,Form,File, APIRouter, UploadFile
from numpy import argmax
from numpy import max
from numpy import array
from fastapi.middleware.cors import CORSMiddleware


#print(tf.version.VERSION)



#model_dir = 'C:\2019-Leon-eigene-Dateien\Studium\6 Semester\Integrationsseminar\Integration\DHBW\model.h5'
#model = tf.keras.load_model(
 #   model_dir,
 #   custom_objects=None, compile=True)



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

@api_router.post("/anfrage/")
async def create_anfrage(file: UploadFile):
    #tempfolder 
    x = preprocessing()
    output = model.predict()
    #anfrage_dict = anfrage.dict()
    #return anfrage_dict
    return {"filename": file.filename}
app.include_router(api_router)