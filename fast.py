import numpy as np
from fastapi import FastAPI
from fastapi import UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from projet.api.gcp_utils import load_model_from_gcp
import cv2

app = FastAPI()

app.state.model = load_model_from_gcp()

# On charge le modèle une seule fois depuis GCP et on le stocke dans app.state.model pour plus de rapidité quand on appellera l'API

# Allowing all middleware is optional, but good practice for dev purposes
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allows all origins
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods
#     allow_headers=["*"], ) # Allows all headers



@app.get("/")
async def root():
    return {"status": "ok"}



@app.post("/predict")
async def predict_image(img: UploadFile = File(...)):
    model = load_model_from_gcp()
    # On traite l'image en la convertissant au bon format
    contents = await img.read()
    img_arr = np.fromstring(contents, np.uint8)
    img_cv2 = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    img_cv2_resized = cv2.resize(img_cv2, (640, 640))

    #On récupère le modèle
    model = app.state.model
    if not model:
        model = app.state.model = load_model_from_gcp()


    #On fait la prédiction
    prediction = model.predict(img_cv2_resized, save=False, imgsz=640, vid_stride=1, conf=0.2)

    # On récupère les infos qui nous intéressent
    boxes = prediction[0].boxes
    waste_categories = boxes.cls.numpy().tolist()
    confidence_score = boxes.conf.numpy().tolist()
    bounding_boxes = boxes.xyxy.numpy().tolist()

    last_prediction = {
        "waste_categories": waste_categories,
        "confidence_score": confidence_score,
        "bounding_boxes": bounding_boxes
    }

    return last_prediction




## Preprocessing image:
# On récupère l'image en bytes
# On la transforme en array
# On la transforme en object qui peut être utilisé par cv2
