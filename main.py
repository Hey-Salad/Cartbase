import os
import torch
from fastapi import FastAPI, File, UploadFile
from google.cloud import storage
from point_e.models.download import load_checkpoint
from point_e.diffusion.sampler import PointCloudSampler
from PIL import Image
from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.models.configs import MODEL_CONFIGS, model_from_config



app = FastAPI()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/asoga/Documents/delta-lore-455512-h7-d20d2824dd02.json"
client = storage.Client()
bucket = client.bucket("recamera_3d_images")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
base_model =  model_from_config(MODEL_CONFIGS["base1B"], device)
#model = load_checkpoint("base1B", device=device)

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

    # Upload to Google Cloud Storage
    blob = bucket.blob(file.filename.replace('.jpg', '.ply'))
    blob.upload_from_filename(model_path)

    return {"message": "3D Model uploaded successfully!", "model_url": blob.public_url}
