import pytest
from httpx import AsyncClient, ASGITransport
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import io
import base64
from unittest.mock import patch, MagicMock
from fastapi import FastAPI
import main 
import matplotlib
print("MATPLOTLIB VERSION:", matplotlib.__version__)


@pytest.fixture(autouse=True)
def mock_gcs():
    with patch("main.storage.Client") as MockStorageClient:
        mock_client_instance = MagicMock()
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        
        MockStorageClient.return_value = mock_client_instance
        mock_client_instance.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob

        mock_blob.upload_from_file.return_value = None
        mock_blob.upload_from_filename.return_value = None
        mock_blob.public_url = "http://fake-url.com/model.ply"

        yield

@pytest.fixture
def test_image():
    image = Image.new("RGB", (224, 224), color=(0, 255, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.mark.asyncio
@patch("main.detect_freshness")
async def test_detect_freshness(mock_detect, test_image):
    mock_detect.return_value = {"freshness": "fresh"}  # Mocked response

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/detect-freshness/",
            files={"file": ("test.jpg", io.BytesIO(test_image), "image/jpeg")}
        )
    assert response.status_code == 200
    assert response.json() == {"freshness": "fresh"}


@pytest.mark.asyncio
@patch("main.detect_freshness_base64")
async def test_detect_freshness_base64(mock_detect, test_image):
    mock_detect.return_value = {"freshness": "fresh"}  # Mocked response

    encoded = base64.b64encode(test_image).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{encoded}"

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/detect-freshness-base64/",
            json={"object": "banana", "image": data_url}
        )
    assert response.status_code == 200
    assert response.json() == {"freshness": "fresh"}


@pytest.mark.asyncio
@patch("main.PointCloudSampler")
@patch("main.DIFFUSION_CONFIGS")
@patch("main.plot_point_cloud")
@patch("main.load_checkpoint")
@patch("main.model_from_config")
@patch("main.diffusion_from_config")
async def test_upload_model_generation(mock_diffusion, mock_model, mock_checkpoint, mock_plot, mock_configs, mock_sampler, test_image):
    # Mock point cloud response
    mock_pc = MagicMock()
    mock_pc.write_ply = MagicMock()
    mock_sampler.return_value = mock_pc  
    mock_plot.return_value = None 
    mock_model.return_value = MagicMock()  
    mock_checkpoint.return_value = MagicMock() 
    mock_configs.return_value = MagicMock()  

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/upload/",
            files={"file": ("test.jpg", io.BytesIO(test_image), "image/jpeg")}
        )

    assert response.status_code == 200
    assert "model_url" in response.json()


@pytest.mark.asyncio
@patch("main.PointCloudSampler")
@patch("main.DIFFUSION_CONFIGS")
@patch("main.plot_point_cloud")
@patch("main.load_checkpoint")
@patch("main.model_from_config")
@patch("main.diffusion_from_config")
async def test_upload_base64_model_generation(mock_diffusion, mock_model, mock_checkpoint, mock_plot, mock_configs, mock_sampler, test_image):
    encoded = base64.b64encode(test_image).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{encoded}"

    # Mock point cloud response
    mock_pc = MagicMock()
    mock_pc.write_ply = MagicMock()
    mock_sampler.return_value = mock_pc  
    mock_plot.return_value = None 
    mock_model.return_value = MagicMock()  
    mock_checkpoint.return_value = MagicMock() 
    mock_configs.return_value = MagicMock()

    transport = ASGITransport(app=main.app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        response = await ac.post(
            "/upload-base64/",
            json={"object": "banana", "image": data_url}
        )

    assert response.status_code == 200
    assert "model_url" in response.json()



    