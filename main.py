import os
import torch
import io
import random
import base64
from fastapi import FastAPI, File, UploadFile
from google.cloud import storage
from pydantic import BaseModel
from point_e.models.download import load_checkpoint
from point_e.diffusion.sampler import PointCloudSampler
from PIL import Image
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from dotenv import load_dotenv



app = FastAPI()

load_dotenv()

credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
print("Credentials path:", credentials_path)

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
client = storage.Client()
bucket = client.bucket("recamera_3d_images")

if torch.backends.mps.is_available():
    device = torch.device("mps")  # Apple GPU
elif torch.cuda.is_available():
    device = torch.device("cuda")  # NVIDIA GPU
else:
    device = torch.device("cpu")  # Fallback
    
print("Using device:", device)
base_model =  model_from_config(MODEL_CONFIGS["base300M"], device)
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['base300M'])


sampler = PointCloudSampler(
    device=device,
    models=[base_model], 
    diffusions=[base_diffusion], 
    num_points=4096, 
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0],
    use_karras=[True],
    karras_steps=[64],
    sigma_min=[1e-3], 
    sigma_max=[120],  
    s_churn=[3],       
    model_kwargs_key_filter=["*"],  
)

@app.post("/upload/")
async def upload_and_generate_3d(file: UploadFile = File(...)):
    image_path = f"temp/{file.filename}"
    with open(image_path, "wb") as img_file:
        img_file.write(await file.read())
    img = Image.open(image_path).convert("RGB")

    with torch.no_grad():
        point_cloud = sampler.sample(img)
    
    # Save the generated 3D model
    model_path = f"temp/{file.filename.replace('.jpg', '.ply')}"
    with open(model_path, "wb") as model_file:
        torch.save(point_cloud, model_file)

    # Upload the 3D model to Google Cloud Storage
    blob = bucket.blob(file.filename.replace('.jpg', '.ply'))
    blob.upload_from_filename(model_path)

    return {"message": "3D Model uploaded successfully!", "model_url": blob.public_url}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Smart Food Inventory API"}

def predict_freshness(image):
    # Mocked freshness prediction
    classes = ["Fresh", "Slightly Spoiled", "Spoiled"]
    return random.choice(classes)

@app.post("/detect-freshness/")
async def detect_freshness(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    freshness_result = predict_freshness(image)

    # Save and upload result to Google Cloud
    result_filename = file.filename.replace(".jpg", "_freshness.txt")
    result_path = f"temp/{result_filename}"

    with open(result_path, "w") as result_file:
        result_file.write(f"Freshness: {freshness_result}\n")

    blob = bucket.blob(result_filename)
    blob.upload_from_filename(result_path)

    return {"freshness": freshness_result, "result_url": blob.public_url}

class Base64Image(BaseModel):
    object: str
    image: str  # Base64 image string, optionally with data URL prefix

@app.post("/upload-base64/")
async def upload_base64_image(data: Base64Image):
    try:
        header, encoded = data.image.split(",", 1) if "," in data.image else ("", data.image)
        image_bytes = base64.b64decode(encoded)

        os.makedirs("temp", exist_ok=True)
        filename = f"temp/{data.object}.jpg"
        with open(filename, "wb") as f:
            f.write(image_bytes)

        img = Image.open(filename).convert("RGB")

        with torch.no_grad():
            point_cloud = sampler.sample(img)

        model_path = filename.replace(".jpg", ".ply")
        with open(model_path, "wb") as model_file:
            torch.save(point_cloud, model_file)

        blob = bucket.blob(model_path.split("/")[-1])
        blob.upload_from_filename(model_path)

        return {"message": "3D Model uploaded from base64 successfully!", "model_url": blob.public_url}

    except Exception as e:
        return {"error": str(e)}