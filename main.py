import os
import io
import random
import base64
import torch
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from google.cloud import storage
from pydantic import BaseModel
from PIL import Image
from dotenv import load_dotenv
from tqdm.auto import tqdm
from point_e.diffusion.sampler import PointCloudSampler
from point_e.diffusion.configs import DIFFUSION_CONFIGS
from point_e.util.plotting import plot_point_cloud
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.models.download import load_checkpoint
from point_e.diffusion.configs import diffusion_from_config
from fastapi.middleware.cors import CORSMiddleware

# --- Initialize FastAPI App ---
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
freshness_model = hub.KerasLayer("https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_b0/classification/2")

freshness_labels = ["Fresh", "Slightly Spoiled", "Spoiled"]

def predict_freshness(image: Image.Image) -> str:
    # Preprocess the image
    image = image.convert("RGB")
    image = image.resize((224, 224))
    image_array = np.array(image).astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # shape: (1, 224, 224, 3)

    # Predict using the model
    predictions = tf.nn.softmax(freshness_model(image_array)).numpy()
    predicted_class = np.argmax(predictions)

    # Map to freshness label (placeholder logic)
    return freshness_labels[predicted_class % len(freshness_labels)]


@app.get("/", response_class=HTMLResponse)
def read_root():
    return FileResponse("static/index.html")

@app.get("/ui", response_class=HTMLResponse)
def serve_ui():
    with open("static/index.html") as f:
        return f.read()

# --- Load Environment and Google Cloud Credentials ---
load_dotenv()
credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
print("Credentials path:", credentials_path)

client = storage.Client()
bucket = client.bucket("recamera_3d_images")
os.makedirs("temp", exist_ok=True)

# --- Device Setup ---
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print("Using device:", device)

# --- Load Point-E Models ---
base_name = 'base40M-imagevec'

print('Loading base model...')
base_model = model_from_config(MODEL_CONFIGS[base_name], device)
base_model.load_state_dict(load_checkpoint(base_name, device))
base_model.eval()
base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('Loading upsampler model...')
upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
upsampler_model.load_state_dict(load_checkpoint('upsample', device))
upsampler_model.eval()
upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

# --- Point Cloud Sampler ---
sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 3.0],
)

# --- Upload Image and Generate 3D Point Cloud (.ply) ---
@app.post("/upload/")
async def upload_and_generate_3d(file: UploadFile = File(...)):
    try:
        image_path = f"temp/{file.filename}"
        with open(image_path, "wb") as img_file:
            img_file.write(await file.read())

        print(f"Saved image to {image_path}")

        with Image.open(image_path) as img:
            img = img.convert("RGB")

            print("Starting point cloud generation...")
            samples = None
            for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
                samples = x

        print("Sampling done, converting to point cloud...")
        pc_list = sampler.output_to_point_clouds(samples)
        if not pc_list:
            raise ValueError("Point cloud conversion failed, got empty list")

        pc = pc_list[0]

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        model_path = f"temp/{base_name}.ply"

        with open(model_path, "wb") as f:
            pc.write_ply(f)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} was not created.")
        if os.path.getsize(model_path) == 0:
            raise ValueError(f"{model_path} is empty. Model generation may have failed.")

        print(f"Uploading {model_path} to GCS as {os.path.basename(model_path)}")
        blob = bucket.blob(os.path.basename(model_path))
        blob.upload_from_filename(model_path)
        print("Upload to GCS successful!")

        return {
            "message": "3D Model uploaded successfully!",
            "model_url": blob.public_url
        }

    except Exception as e:
        print(f"🚨 Error during upload: {str(e)}")
        return {"error": str(e)}



# --- Mock Freshness Detection ---
# def predict_freshness(image):
#    classes = ["Fresh", "Slightly Spoiled", "Spoiled"]
#    return random.choice(classes)

@app.post("/detect-freshness/")
async def detect_freshness(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    freshness_result = predict_freshness(image)

    result_filename = file.filename.replace(".jpg", "_freshness.txt")
    result_path = f"temp/{result_filename}"
    with open(result_path, "w") as result_file:
        result_file.write(f"Freshness: {freshness_result}\n")

    blob = bucket.blob(result_filename)
    blob.upload_from_filename(result_path)

    return {"freshness": freshness_result, "result_url": blob.public_url}

# --- Base64 Image Upload and 3D Model Generation ---
class Base64Image(BaseModel):
    object: str
    image: str  # Base64 image string

@app.post("/upload-base64/")
async def upload_base64_image(data: Base64Image):
    try:
        # Decode the base64 image data
        header, encoded = data.image.split(",", 1) if "," in data.image else ("", data.image)
        image_bytes = base64.b64decode(encoded)

        # Save the image to a temporary file
        filename = f"temp/{data.object}.jpg"
        with open(filename, "wb") as f:
            f.write(image_bytes)

        # Open and convert image
        with Image.open(filename) as img:
            img = img.convert("RGB")

            # Sample 3D point cloud
            samples = None
            for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(images=[img]))):
                samples = x

        # Extract point cloud and save to .ply
        pc = sampler.output_to_point_clouds(samples)[0]
        fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75), (0.75, 0.75, 0.75)))

        # Create the model path
        model_path = f"temp/{data.object}.ply"
        with open(model_path, "wb") as f:
            pc.write_ply(f)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_path} was not created.")
        if os.path.getsize(model_path) == 0:
            raise ValueError(f"{model_path} is empty. Model generation may have failed.")

        print(f"Uploading {model_path} to GCS as {os.path.basename(model_path)}")
        blob = bucket.blob(os.path.basename(model_path))
        blob.upload_from_filename(model_path)
        print("Upload to GCS successful!")

        return {
            "message": "3D Model uploaded successfully!",
            "model_url": blob.public_url
        }

    except Exception as e:
        print(f"🚨 Error during upload: {str(e)}")
        return {"error": str(e)}

@app.post("/detect-freshness-base64/")
async def detect_freshness_base64(data: Base64Image):
    try:
        header, encoded = data.image.split(",", 1) if "," in data.image else ("", data.image)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        freshness_result = predict_freshness(image)

        result_filename = f"{data.object}_freshness.txt"
        result_path = f"temp/{result_filename}"
        with open(result_path, "w") as result_file:
            result_file.write(f"Freshness: {freshness_result}\n")

        blob = bucket.blob(result_filename)
        blob.upload_from_filename(result_path)

        return {"freshness": freshness_result, "result_url": blob.public_url}
    except Exception as e:
        return {"error": str(e)}