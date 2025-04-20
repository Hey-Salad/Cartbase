import pytest
from httpx import AsyncClient
from PIL import Image
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import io
import base64
from unittest.mock import patch, MagicMock

import main 

@pytest.fixture
def test_image():
    image = Image.new("RGB", (224, 224), color=(0, 255, 0))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_detect_freshness(test_image):
    async with AsyncClient(app=main, base_url="http://test") as ac:
        response = await ac.post(
            "/detect-freshness/",
            files={"file": ("test.jpg", io.BytesIO(test_image), "image/jpeg")}
        )
    assert response.status_code == 200
    assert "freshness" in response.json()


@pytest.mark.asyncio
async def test_detect_freshness_base64(test_image):
    encoded = base64.b64encode(test_image).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{encoded}"

    async with AsyncClient(app=main, base_url="http://test") as ac:
        response = await ac.post(
            "/detect-freshness-base64/",
            json={"object": "banana", "image": data_url}
        )
    assert response.status_code == 200
    assert "freshness" in response.json()


@pytest.mark.asyncio
@patch("main.sampler.sample_batch_progressive")
@patch("main.sampler.output_to_point_clouds")
@patch("main.bucket.blob")
async def test_upload_model_generation(mock_blob, mock_output_pc, mock_sample, test_image):
    # Mock point cloud response
    mock_pc = MagicMock()
    mock_pc.write_ply = MagicMock()
    mock_output_pc.return_value = [mock_pc]
    mock_sample.return_value = [{"dummy": "data"}]
    mock_blob.return_value.upload_from_filename = MagicMock()
    mock_blob.return_value.public_url = "http://fake-url.com/model.ply"

    async with AsyncClient(app=main, base_url="http://test") as ac:
        response = await ac.post(
            "/upload/",
            files={"file": ("test.jpg", io.BytesIO(test_image), "image/jpeg")}
        )

    assert response.status_code == 200
    assert "model_url" in response.json()


@pytest.mark.asyncio
@patch("main.sampler.sample_batch_progressive")
@patch("main.sampler.output_to_point_clouds")
@patch("main.bucket.blob")
async def test_upload_base64_model_generation(mock_blob, mock_output_pc, mock_sample, test_image):
    encoded = base64.b64encode(test_image).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{encoded}"

    # Mock point cloud response
    mock_pc = MagicMock()
    mock_pc.write_ply = MagicMock()
    mock_output_pc.return_value = [mock_pc]
    mock_sample.return_value = [{"dummy": "data"}]
    mock_blob.return_value.upload_from_filename = MagicMock()
    mock_blob.return_value.public_url = "http://fake-url.com/model.ply"

    async with AsyncClient(app=main, base_url="http://test") as ac:
        response = await ac.post(
            "/upload-base64/",
            json={"object": "banana", "image": data_url}
        )

    assert response.status_code == 200
    assert "model_url" in response.json()



    