import os
import uuid
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
import shutil
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    logger.warning("OpenAI API key not found. Using mock AI suggestions.")

# Initialize FastAPI and OpenAI client
app = FastAPI(title="NaNja Backend", description="API for NaNja data preprocessing tool")
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Temporary storage directories
UPLOAD_DIR = Path("uploads")
CLEANED_DIR = Path("cleaned")
for dir_path in [UPLOAD_DIR, CLEANED_DIR]:
    dir_path.mkdir(exist_ok=True)

# Pydantic models
class ColumnSummary(BaseModel):
    name: str
    type: str
    nulls: int
    unique: int

class DatasetSummary(BaseModel):
    rows: int
    duplicates: int
    columns: List[ColumnSummary]

class CleaningOptions(BaseModel):
    removeDuplicates: bool = False
    dropMissing: bool = False
    fillMissing: bool = False
    fillMethod: Optional[str] = None

class Suggestion(BaseModel):
    id: int
    text: str
    type: str
    column: Optional[str] = None

class SuggestionsResponse(BaseModel):
    suggestions: List[Suggestion]

class DatasetPreview(BaseModel):
    columns: List[str]
    data: List[List]

# Helper functions
async def read_file(file_path: Path) -> pd.DataFrame:
    try:
        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        elif file_path.suffix.lower() in [".xlsx", ".xls"]:
            return pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
    except Exception as e:
        logger.error(f"Error reading file {file_path}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error reading file: {str(e)}")

def get_dataset_summary(df: pd.DataFrame) -> Dict:
    return {
        "rows": len(df),
        "duplicates": len(df) - len(df.drop_duplicates()),
        "columns": [
            {
                "name": col,
                "type": str(df[col].dtype),
                "nulls": int(df[col].isnull().sum()),
                "unique": int(df[col].nunique())
            }
            for col in df.columns
        ]
    }

async def get_ai_suggestions(summary: Dict) -> List[Dict]:
    if not openai_client:
        return [
            {"id": 1, "text": "Drop columns with >50% missing values", "type": "drop"},
            {"id": 2, "text": "Create categorical bins for numerical columns", "type": "feature"},
            {"id": 3, "text": "Normalize numerical columns for ML", "type": "transform"}
        ]
    
    prompt = f"""
You are an expert data scientist. Analyze the following dataset summary and provide 3-5 actionable suggestions for data preprocessing to improve machine learning model performance. Suggestions can include dropping columns, feature engineering, or data transformations. For each suggestion, specify the target column if applicable. Format each suggestion as a JSON object with 'id', 'text', 'type' ('drop', 'feature', or 'transform'), and optional 'column'.

Dataset Summary:
- Rows: {summary['rows']}
- Duplicates: {summary['duplicates']}
- Columns: {summary['columns']}

Provide suggestions in JSON format: {{"suggestions": [...]}}
"""
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        suggestions = eval(response.choices[0].message.content).get("suggestions", [])
        return suggestions
    except Exception as e:
        logger.error(f"Error getting AI suggestions: {str(e)}")
        return [
            {"id": 1, "text": "Error fetching AI suggestions; drop high-null columns", "type": "drop"}
        ]

def apply_cleaning(df: pd.DataFrame, options: CleaningOptions) -> pd.DataFrame:
    df_clean = df.copy()
    try:
        if options.removeDuplicates:
            df_clean = df_clean.drop_duplicates()
        if options.dropMissing:
            df_clean = df_clean.dropna()
        if options.fillMissing and options.fillMethod:
            for col in df_clean.columns:
                if df_clean[col].isnull().any():
                    if options.fillMethod == "mean" and pd.api.types.is_numeric_dtype(df_clean[col]):
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                    elif options.fillMethod == "median" and pd.api.types.is_numeric_dtype(df_clean[col]):
                        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                    elif options.fillMethod == "mode":
                        df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
    except Exception as e:
        logger.error(f"Error applying cleaning: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error applying cleaning: {str(e)}")
    return df_clean

def apply_suggestion(df: pd.DataFrame, suggestion: Dict) -> pd.DataFrame:
    df_clean = df.copy()
    try:
        if suggestion["type"] == "drop" and suggestion.get("column"):
            df_clean = df_clean.drop(columns=[suggestion["column"]])
        elif suggestion["type"] == "feature" and suggestion.get("column"):
            if pd.api.types.is_numeric_dtype(df_clean[suggestion["column"]]):
                df_clean[f"{suggestion['column']}_binned"] = pd.qcut(df_clean[suggestion["column"]], q=4, duplicates="drop")
        elif suggestion["type"] == "transform" and suggestion.get("column"):
            if pd.api.types.is_numeric_dtype(df_clean[suggestion["column"]]):
                df_clean[suggestion["column"]] = (df_clean[suggestion["column"]] - df_clean[suggestion["column"]].min()) / (df_clean[suggestion["column"]].max() - df_clean[suggestion["column"]].min())
    except Exception as e:
        logger.error(f"Error applying suggestion {suggestion}: {str(e)}")
    return df_clean

# API Endpoints
@app.post("/api/upload", response_model=Dict[str, str])
async def upload_file(file: UploadFile = File(...)):
    if file.content_type not in ["text/csv", "application/vnd.ms-excel", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV and Excel are supported.")
    
    file_id = str(uuid.uuid4())
    file_path = UPLOAD_DIR / f"{file_id}{Path(file.filename).suffix}"
    
    try:
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {"file_id": file_id}
    except Exception as e:
        logger.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")

@app.get("/api/summary/{file_id}", response_model=DatasetSummary)
async def get_summary(file_id: str):
    file_path = next(UPLOAD_DIR.glob(f"{file_id}.*"), None)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    df = await read_file(file_path)
    summary = get_dataset_summary(df)
    return summary

@app.get("/api/preview/{file_id}", response_model=DatasetPreview)
async def get_preview(file_id: str):
    file_path = next(UPLOAD_DIR.glob(f"{file_id}.*"), None)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    df = await read_file(file_path)
    max_rows = 100
    preview_df = df.head(max_rows)
    columns = preview_df.columns.tolist()
    data = preview_df.fillna("").astype(str).values.tolist()
    return {"columns": columns, "data": data}

@app.post("/api/clean/{file_id}", response_model=Dict[str, str])
async def clean_dataset(file_id: str, options: CleaningOptions):
    file_path = next(UPLOAD_DIR.glob(f"{file_id}.*"), None)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    df = await read_file(file_path)
    df_clean = apply_cleaning(df, options)
    
    cleaned_file_id = str(uuid.uuid4())
    cleaned_path = CLEANED_DIR / f"{cleaned_file_id}.csv"
    df_clean.to_csv(cleaned_path, index=False)
    
    return {"cleaned_file_id": cleaned_file_id}

@app.get("/api/suggestions/{file_id}", response_model=SuggestionsResponse)
async def get_suggestions(file_id: str):
    file_path = next(UPLOAD_DIR.glob(f"{file_id}.*"), None)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    df = await read_file(file_path)
    summary = get_dataset_summary(df)
    suggestions = await get_ai_suggestions(summary)
    return {"suggestions": suggestions}

@app.post("/api/apply-suggestions/{file_id}", response_model=Dict[str, str])
async def apply_suggestions(file_id: str, suggestion_ids: List[int]):
    file_path = next(UPLOAD_DIR.glob(f"{file_id}.*"), None)
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    df = await read_file(file_path)
    suggestions = (await get_suggestions(file_id)).suggestions
    df_clean = df.copy()
    
    for sug_id in suggestion_ids:
        suggestion = next((s for s in suggestions if s.id == sug_id), None)
        if suggestion:
            df_clean = apply_suggestion(df_clean, suggestion.dict())
    
    cleaned_file_id = str(uuid.uuid4())
    cleaned_path = CLEANED_DIR / f"{cleaned_file_id}.csv"
    df_clean.to_csv(cleaned_path, index=False)
    
    return {"cleaned_file_id": cleaned_file_id}

@app.get("/api/download/{cleaned_file_id}")
async def download_file(cleaned_file_id: str):
    file_path = CLEANED_DIR / f"{cleaned_file_id}.csv"
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Cleaned file not found")
    
    return FileResponse(file_path, filename="cleaned_dataset.csv", media_type="text/csv")