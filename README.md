
# Smart Food Inventory System - Backend Documentation

## Overview

This FastAPI-based backend powers a Smart Food Inventory System that captures images from a reCamera (via Node-RED), processes them into 3D point clouds using OpenAI's Point-E model, and analyzes the freshness of food using a TensorFlow model from TensorFlow Hub. The resulting data is stored in Google Cloud Storage and served via RESTful endpoints.
To set up, create a new bucket for where you want the images and data to be uploaded and save the resulting json credentials in a secure directory. You can change main.py to point your credentials to that json file wherever you have it saved.  

[reCamera Startup Guide](https://wiki.seeedstudio.com/recamera_getting_started/)
## Features

- **Image Upload & Storage**: Receives image uploads and saves them to a local `temp` directory and Google Cloud Storage.
- **3D Model Generation**: Converts 2D images into 3D point clouds using Point-E.
- **Freshness Detection**: Classifies food freshness using a pre-trained MobileNetV2 TensorFlow model.
- **Base64 Support**: Accepts both file uploads and base64-encoded image submissions.
- **Frontend UI Support**: Serves a static HTML frontend for quick access and testing.

## API Endpoints

### GET `/`
Serves the static `index.html` from the `static/` directory.

### GET `/ui`
Same as `/`, explicitly serves the UI for testing or visualization.

### POST `/upload/`
Receives an uploaded image, generates a 3D point cloud, and uploads the `.ply` file to GCS.

**Request**:
- `file`: Image file (`multipart/form-data`)

**Response**:
- `model_url`: Public GCS link to the generated `.ply` file

### POST `/detect-freshness/`
Runs freshness prediction on an uploaded image.

**Request**:
- `file`: Image file (`multipart/form-data`)

**Response**:
- `freshness`: Detected freshness category
- `result_url`: Public GCS link to the `.txt` report

### POST `/upload-base64/`
Accepts a base64-encoded image, generates a 3D model, and uploads it.

**Request** (JSON):
```json
{
  "object": "banana",
  "image": "data:image/jpeg;base64,..."
}
```

**Response**:
- `model_url`: Public GCS link to the `.ply` file

### POST `/detect-freshness-base64/`
Accepts a base64-encoded image and predicts freshness.

**Request** (JSON):
```json
{
  "object": "banana",
  "image": "data:image/jpeg;base64,..."
}
```

**Response**:
- `freshness`: Predicted freshness label
- `result_url`: Link to result text file

## Technologies Used

- **FastAPI**: Backend framework
- **Torch + Point-E**: 3D model generation
- **TensorFlow Hub**: Freshness classification
- **Google Cloud Storage**: Persistent storage
- **Vue.js (Frontend)**: UI showing live updates from 3D modeling process.

## File Structure Highlights

- `static/`: Contains `index.html` UI
- `temp/`: Temporary storage for uploaded images and generated models
- `.env`: Stores GCS credentials path
- `main.py`: Main FastAPI app

## Future Enhancements
- Improve freshness classification with custom fine-tuned models
- Add logging and analytics dashboard
- Streamline frontend with live camera feed and real-time status

---

*Last updated: April 2025*
