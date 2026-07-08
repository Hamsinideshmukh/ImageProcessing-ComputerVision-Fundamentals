from fastapi import FastAPI
from fastapi import UploadFile,File
from fastapi.responses import StreamingResponse
import cv2
import numpy as np
import io
from transformers import pipeline
from PIL import Image

classifier = pipeline(
    "image-classification",
    model = "google/vit-base-patch16-224"
)

app = FastAPI()

@app.get("/")
def root():
    return {"status":"Server is running"}

@app.post("/process-image")
async def process_image(file:UploadFile = File(...)):
    # Read the uploaded image bytes 
    image_bytes = await file.read()
    # Convert bytes to numpy array
    np_array = np.frombuffer(image_bytes, np.uint8)
    # Decode image 
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    # Image processing
    gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)
    # Encode image back to png
    _, encoded = cv2.imencode(".png",gray)
    return StreamingResponse(
        io.BytesIO(encoded.tobytes()),
        media_type = "image/png"
    )

@app.post("/classify")
async def classify_image(file:UploadFile = File(...)):
    image_bytes = await file.read()

    np_array = np.frombuffer(image_bytes, np.unit8)

    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    result = classifier(pil_image)
    return result