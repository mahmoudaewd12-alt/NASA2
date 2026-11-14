"""
Landing Suitability API
تشغيل:  pip install fastapi uvicorn joblib scikit-learn pandas
         uvicorn api:app --reload --port 8000
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import joblib, pandas as pd

bundle = joblib.load("landing_model.joblib")
app = FastAPI(title="Lunar Landing Suitability")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class SiteInput(BaseModel):
    sun_elevation_deg: float = Field(..., ge=-90, le=90)           # من الفرونت إند (سلايدر/إدخال)
    illumination_percent: float = Field(..., ge=0, le=100)
    earth_visibility_percent: float = Field(..., ge=0, le=100)
    sun_azimuth_deg: float = 180.0
    latitude_deg: float = 0.0
    east_longitude_deg: float = 0.0
    slope_deg: Optional[float] = None      # غير مستخدم حاليًا في الموديل
    elevation_m: Optional[float] = None    # غير مستخدم حاليًا في الموديل

@app.post("/predict")
def predict(inp: SiteInput):
    earth_visible = int(inp.earth_visibility_percent > 50)
    x = pd.DataFrame([[inp.sun_elevation_deg, inp.sun_azimuth_deg, inp.latitude_deg, inp.east_longitude_deg,
                       inp.illumination_percent, inp.earth_visibility_percent, earth_visible]],
                     columns=bundle["features"])
    score = float(bundle["regressor"].predict(x)[0])
    cls = str(bundle["classifier"].predict(x)[0])
    return {"landing_score": round(score, 1), "landing_class": cls,
            "suitable": cls in ("Excellent", "Good")}



