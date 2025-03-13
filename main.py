from fastapi import FastAPI, File, UploadFile, HTTPException
from google.cloud import storage
import os



app = FastAPI()