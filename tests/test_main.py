import pytest
import pandas as pd
from fastapi.testclient import TestClient
from backend.main import app, UPLOAD_DIR, CLEANED_DIR
from unittest.mock import patch
import os
import shutil
from pathlib import Path
import uuid

# Initialize test client
client = TestClient(app)

# Setup and teardown for tests
@pytest.fixture(autouse=True)
def setup_and_teardown():
    # Create temporary directories
    UPLOAD_DIR.mkdir(exist_ok=True)
    CLEANED_DIR.mkdir(exist_ok=True)
    yield
    # Clean up directories after each test
    shutil.rmtree(UPLOAD_DIR, ignore_errors=True)
    shutil.rmtree(CLEANED_DIR, ignore_errors=True)

@pytest.fixture
def sample_csv_path():
    # Path to test dataset
    return Path("test_datasets/Data.csv")

@pytest.fixture
def mock_groq():
    # Mock Groq LLM responses
    with patch("backend.main.llm_client") as mock_client:
        mock_completion = {
            "choices": [{"message": {"content": '{"suggestions": [{"id": 1, "text": "Drop column age due to high nulls", "type": "drop", "column": "age"}]}'}}]
        }
        mock_client.chat.completions.create.return_value = type("Mock", (), mock_completion)
        yield mock_client

def test_upload_file(sample_csv_path):
    # Test file upload endpoint
    with open(sample_csv_path, "rb") as f:
        response = client.post("/api/upload", files={"file": ("Data.csv", f, "text/csv")})
    assert response.status_code == 200
    assert "file_id" in response.json()
    file_id = response.json()["file_id"]
    assert any(UPLOAD_DIR.glob(f"{file_id}.csv"))

def test_upload_invalid_file():
    # Test uploading unsupported file type
    with open("test.txt", "w") as f:
        f.write("Invalid content")
    with open("test.txt", "rb") as f:
        response = client.post("/api/upload", files={"file": ("test.txt", f, "text/plain")})
    assert response.status_code == 400
    assert "Invalid file type" in response.json()["detail"]
    os.remove("test.txt")

def test_get_summary(sample_csv_path):
    # Test dataset summary endpoint
    with open(sample_csv_path, "rb") as f:
        upload_response = client.post("/api/upload", files={"file": ("Data.csv", f, "text/csv")})
    file_id = upload_response.json()["file_id"]
    
    response = client.get(f"/api/summary/{file_id}")
    assert response.status_code == 200
    summary = response.json()
    assert "rows" in summary
    assert "duplicates" in summary
    assert "columns" in summary
    assert isinstance(summary["columns"], list)
    assert all(key in summary["columns"][0] for key in ["name", "type", "nulls", "unique"])

def test_get_summary_invalid_file_id():
    # Test summary with invalid file_id
    response = client.get(f"/api/summary/{uuid.uuid4()}")
    assert response.status_code == 404
    assert "File not found" in response.json()["detail"]

def test_get_preview(sample_csv_path):
    # Test dataset preview endpoint
    with open(sample_csv_path, "rb") as f:
        upload_response = client.post("/api/upload", files={"file": ("Data.csv", f, "text/csv")})
    file_id = upload_response.json()["file_id"]
    
    response = client.get(f"/api/preview/{file_id}")
    assert response.status_code == 200
    preview = response.json()
    assert preview["preview_type"] == "original"
    assert "columns" in preview
    assert "data" in preview
    assert isinstance(preview["columns"], list)
    assert isinstance(preview["data"], list)

def test_clean_dataset(sample_csv_path):
    # Test basic cleaning endpoint
    with open(sample_csv_path, "rb") as f:
        upload_response = client.post("/api/upload", files={"file": ("Data.csv", f, "text/csv")})
    file_id = upload_response.json()["file_id"]
    
    cleaning_options = {
        "removeDuplicates": True,
        "dropMissing": False,
        "fillMissing": False,
        "fillMethod": None
    }
    response = client.post(f"/api/clean/{file_id}", json=cleaning_options)
    assert response.status_code == 200
    assert "cleaned_file_id" in response.json()
    cleaned_file_id = response.json()["cleaned_file_id"]
    assert (CLEANED_DIR / f"{cleaned_file_id}.csv").exists()

def test_get_suggestions(mock_groq, sample_csv_path):
    # Test AI suggestions endpoint
    with open(sample_csv_path, "rb") as f:
        upload_response = client.post("/api/upload", files={"file": ("Data.csv", f, "text/csv")})
    file_id = upload_response.json()["file_id"]
    
    response = client.get(f"/api/suggestions/{file_id}")
    assert response.status_code == 200
    suggestions = response.json()["suggestions"]
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0
    assert all(key in suggestions[0] for key in ["id", "text", "type", "column"])
    assert suggestions[0]["text"] == "Drop column age due to high nulls"

def test_apply_suggestions(mock_groq, sample_csv_path):
    # Test applying AI suggestions
    with open(sample_csv_path, "rb") as f:
        upload_response = client.post("/api/upload", files={"file": ("Data.csv", f, "text/csv")})
    file_id = upload_response.json()["file_id"]
    
    # Generate suggestions
    client.get(f"/api/suggestions/{file_id}")
    
    # Apply suggestion ID 1
    response = client.post(f"/api/apply-suggestions/{file_id}", json=[1])
    assert response.status_code == 200
    result = response.json()
    assert "cleaned_file_id" in result
    assert "applied_suggestions" in result
    assert "failed_suggestions" in result
    assert len(result["applied_suggestions"]) > 0
    assert (CLEANED_DIR / f"{result['cleaned_file_id']}.csv").exists()

def test_download_file(mock_groq, sample_csv_path):
    # Test downloading cleaned dataset
    with open(sample_csv_path, "rb") as f:
        upload_response = client.post("/api/upload", files={"file": ("Data.csv", f, "text/csv")})
    file_id = upload_response.json()["file_id"]
    
    # Apply cleaning to generate a cleaned file
    cleaning_options = {"removeDuplicates": True}
    clean_response = client.post(f"/api/clean/{file_id}", json=cleaning_options)
    cleaned_file_id = clean_response.json()["cleaned_file_id"]
    
    response = client.get(f"/api/download/{cleaned_file_id}")
    assert response.status_code == 200
    assert response.headers["content-type"] == "text/csv"
    assert "cleaned_dataset.csv" in response.headers["content-disposition"]

def test_download_invalid_file():
    # Test downloading non-existent cleaned file
    response = client.get(f"/api/download/{uuid.uuid4()}")
    assert response.status_code == 404
    assert "Cleaned file not found" in response.json()["detail"]