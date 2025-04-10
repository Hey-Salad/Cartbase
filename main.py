import os
import ssl
import time
import base64
import io
import random

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage

from point_e.models.download import load_checkpoint
from point_e.diffusion.sampler import PointCloudSampler
from PIL import Image
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from dotenv import load_dotenv

app = FastAPI()
load_dotenv()

# Ensure the temporary folder exists
os.makedirs("temp", exist_ok=True)

# SSL workaround - disable certificate verification
os.environ['PYTHONHTTPSVERIFY'] = '0'
ssl._create_default_https_context = ssl._create_unverified_context

# Set up Google Cloud credentials
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
print("Credentials path:", credentials_path)
if not credentials_path:
    raise RuntimeError("GOOGLE_APPLICATION_CREDENTIALS not set in environment.")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
client = storage.Client()
bucket = client.bucket("recamera_3d_images")

# Set device and prepare models/diffusions for a two-stage pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model Prep for base model
base_name = "base1B"  # You can also try "base300M" for faster but lower quality results
print("Creating base model...")
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])
print("Downloading base checkpoint...")
base_model.load_state_dict(load_checkpoint(base_name, device))

# Model Prep for upsampler model
print("Creating upsample model...")
upsampler_model = model_from_config(MODEL_CONFIGS["upsample"], device)
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS["upsample"])
print("Downloading upsampler checkpoint...")
upsampler_model.load_state_dict(load_checkpoint("upsample", device))

# Instantiate the sampler with two models. Note all parameters must be lists of length 2.
sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 3.0],
    use_karras=[True, True],
    karras_steps=[64, 64],
    sigma_min=[1e-3, 1e-3],
    sigma_max=[120, 160],
    s_churn=[3, 0],
)

# Pydantic model for incoming JSON payload
class FruitImage(BaseModel):
    object: str
    image: str  # Base64 encoded image string

def predict_freshness(image: Image.Image) -> str:
    # Mocked freshness prediction
    classes = ["Fresh", "Slightly Spoiled", "Spoiled"]
    return random.choice(classes)

@app.get("/")
def read_root():
    return {"message": "Welcome to the Smart Food Inventory API"}

@app.post("/upload/")
async def upload_and_generate_3d(data: FruitImage):
    try:
        # Decode Base64 image string
        try:
            image_data = base64.b64decode(data.image)
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid Base64 string.")

        # Save the image to a temporary file
        filename = f"{data.object}_{int(time.time())}.jpg"
        image_path = os.path.join("temp", filename)
        with open(image_path, "wb") as img_file:
            img_file.write(image_data)

        # Open the image with PIL
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid image file.")

        # Pass the PIL image in a list as required by the sampler (model_kwargs expects a list)
        with torch.no_grad():
            model_kwargs = {"images": [img]}
            # Iterate over progressive sampling (as in the original Point-E example)
            samples = None
            for x in sampler.sample_batch_progressive(batch_size=1, model_kwargs=model_kwargs):
                samples = x
            if samples is None:
                raise HTTPException(status_code=500, detail="Sampling failed.")
            point_cloud = sampler.output_to_point_clouds(samples)[0]

        # Save the generated 3D model (point cloud) as a .ply file
        base_name_no_ext, _ = os.path.splitext(filename)
        model_filename = base_name_no_ext + ".ply"
        model_path = os.path.join("temp", model_filename)
        with open(model_path, "wb") as model_file:
            torch.save(point_cloud, model_file)

        # Upload the .ply file to Google Cloud Storage
        blob = bucket.blob(model_filename)
        blob.upload_from_filename(model_path)

        return {"message": "3D Model uploaded successfully!", "model_url": blob.public_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect-freshness/")
async def detect_freshness(data: FruitImage):
    try:
        # Decode Base64 image string
        try:
            image_data = base64.b64decode(data.image)
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid Base64 string.")

        # Generate filename using object label and timestamp
        filename = f"{data.object}_{int(time.time())}.jpg"
        # Open image from decoded bytes
        try:
            img = Image.open(io.BytesIO(image_data)).convert("RGB")
        except Exception:
            raise HTTPException(status_code=422, detail="Invalid image file.")

        # Get a mocked freshness prediction
        freshness_result = predict_freshness(img)

        # Save the freshness result to a text file
        base_name_no_ext, _ = os.path.splitext(filename)
        result_filename = base_name_no_ext + "_freshness.txt"
        result_path = os.path.join("temp", result_filename)
        with open(result_path, "w") as result_file:
            result_file.write(f"Freshness: {freshness_result}\n")

        # Upload the result file to Google Cloud Storage
        blob = bucket.blob(result_filename)
        blob.upload_from_filename(result_path)

        return {"freshness": freshness_result, "result_url": blob.public_url}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
