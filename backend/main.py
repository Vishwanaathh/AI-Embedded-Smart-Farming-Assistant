from fastapi import FastAPI
import joblib
import cv2
import numpy as np
from ultralytics import YOLO
import winsound
import time 
from pydantic import BaseModel
class FertilizerParams(BaseModel):
    temp:float
    hum:float
    moist:float
    crop:str
    soil:str

class CropParams(BaseModel):
    n:float
    p:float
    k:float
    temp:float
    hum:float
    ph:float
    rainfall:float

class IrrigationParams(BaseModel):
    temp:float
    hum:float
    ph:float
    rainfall:float
    moist:float



app=FastAPI()
print("Loading Fertilizer model")
fert = joblib.load("../models/fertilizer_recommendation_model.joblib")
print("Model Loaded")
print("Loading irrigation and ph adjustment model")
irph = joblib.load("../models/irrigation_ph_recommender.pkl")
print("Model Loaded")
print("Loading crop recommender")
crop = joblib.load("../models/crop_recommendation_lgbm.pkl")
print("Model loaded")

@app.get("/")
def root():
    return "Welcome To SMAIRT Farming"


@app.post("/fertilizerrecommend")
def fertt(data:FertilizerParams):
    out=fert.predict([[data.temp,data.hum,data.moist,data.crop,data.soil]])
    return {"fertilizer": str(out[0])}

@app.post("/croprecommend")
def cropp(data:CropParams):
    out=crop.predict([[data.n,data.p,data.k,data.temp,data.hum,data.ph,data.rainfall]])
    return {"crop to grow":str(out)}

@app.post("/irrigationandphcorrection")
def irrphh(data:IrrigationParams):
    out=irph.predict([[data.temp,data.hum,data.ph,data.rainfall,data.moist]])
    return {"irrcorrection":str(out[0][0]),"phcorrection":str(out[0][1])}
    

