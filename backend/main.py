import os
import uuid
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
from groq import Groq
from dotenv import load_dotenv
import shutil
from pathlib import Path
import logging
import json
import re

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logger.warning("GROQ API key not found. Using mock AI suggestions.")

# Initialize FastAPI and Groq client
app = FastAPI(title="NaNja Backend", description="API for NaNja data preprocessing tool")
llm_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

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
    preview_type: str
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

def get_dataset_summary(df: pd.DataFrame, target_column: Optional[str] = None) -> Dict:
    # Calculate null percentages for consistency with Colab
    null_percentages = (df.isnull().sum() / len(df) * 100).round(2).astype(str).to_dict()
    # Get a sample row, convert to dict, handle NaN by converting to string
    sample_row = df.sample(1).iloc[0].fillna("None").astype(str).to_dict()
    
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
        ],
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_percentages": null_percentages,
        "sample_row": sample_row,
        "target": target_column if target_column in df.columns else "None",
        "columns_list": df.columns.tolist()  # Add columns list for suggestion parsing
    }

async def get_ai_suggestions(summary: Dict) -> List[Dict]:
    if not llm_client:
        logger.warning("No LLM client available, returning mock suggestions")
        return [
            {"id": 1, "text": "Drop columns with >50% missing values", "type": "drop"},
            {"id": 2, "text": "Create categorical bins for numerical columns", "type": "feature"},
            {"id": 3, "text": "Normalize numerical columns for ML", "type": "transform"}
        ]
    
    prompt = f"""
I have the following dataset summary:

Columns and types: {summary["dtypes"]}
Null percentages: {summary["null_percentages"]}
Sample row: {summary["sample_row"]}
Target column: {summary["target"]}

Please suggest preprocessing steps to make this dataset suitable for machine learning training.
Suggestions may include: columns to drop, null handling, feature engineering, and encoding advice.

Return the suggestions as a JSON object in the format: {{"suggestions": [{{"id": 1, "text": "Drop column X", "type": "drop", "column": "X"}}, ...]}}
Each suggestion must have an 'id' (integer), 'text' (description), 'type' ('drop', 'feature', or 'transform'), and optional 'column' (target column name).
If you cannot provide suggestions in JSON format, provide them in bullet points (e.g., - Drop column X).
"""
    try:
        logger.info("Attempting to call Groq API...")
        chat_completion = llm_client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
            stream=False,
        )
        logger.info("Groq API call successful")
        response = chat_completion.choices[0].message.content
        logger.info(f"Raw LLM response: {response}")
        
        # Preprocess the response to remove Markdown code block syntax
        # Remove ```json and ``` (or any ``` block)
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[len("```json"):].strip()
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[len("```"):].strip()
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-len("```")].strip()
        logger.info(f"Cleaned LLM response: {cleaned_response}")
        
        # Try to parse as JSON first
        try:
            suggestions = json.loads(cleaned_response).get("suggestions", [])
            logger.info(f"Parsed JSON suggestions: {suggestions}")
            if not suggestions:
                logger.warning("No suggestions found in JSON response")
                return [{"id": 1, "text": "No suggestions provided by AI", "type": "drop"}]
            return suggestions
        except json.JSONDecodeError as json_err:
            logger.warning(f"JSON parsing failed: {str(json_err)}")
            # Fallback to parsing bullet points
            suggestions = []
            lines = cleaned_response.split('\n')
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line.startswith('- ') or line.startswith('* '):
                    text = line[2:].strip()
                    suggestion_type = "drop"  # Default type
                    column = None
                    # Try to infer type and column
                    if "drop" in text.lower():
                        suggestion_type = "drop"
                        column_match = re.search(r"column\s+([a-zA-Z_][a-zA-Z0-9_]*)", text, re.IGNORECASE)
                        column = column_match.group(1) if column_match else None
                    elif "bins" in text.lower() or "feature" in text.lower():
                        suggestion_type = "feature"
                        column_match = re.search(r"for\s+([a-zA-Z_][a-zA-Z0-9_]*)", text, re.IGNORECASE)
                        column = column_match.group(1) if column_match else None
                    elif "normalize" in text.lower() or "encode" in text.lower() or "transform" in text.lower():
                        suggestion_type = "transform"
                        column_match = re.search(r"([a-zA-Z_][a-zA-Z0-9_]*)", text, re.IGNORECASE)
                        column = column_match.group(1) if column_match else None
                    if text:
                        suggestions.append({"id": i, "text": text, "type": suggestion_type, "column": column})
            logger.info(f"Parsed bullet-point suggestions: {suggestions}")
            if not suggestions:
                logger.warning("No valid suggestions parsed from bullet points")
                return [{"id": 1, "text": "Error fetching AI suggestions; drop high-null columns", "type": "drop"}]
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
async def get_preview(file_id: str, cleaned_file_id: Optional[str] = None):
    if cleaned_file_id:
        file_path = CLEANED_DIR / f"{cleaned_file_id}.csv"
        preview_type = "cleaned"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Cleaned file not found")
    else:
        file_path = next(UPLOAD_DIR.glob(f"{file_id}.*"), None)
        preview_type = "original"
        if not file_path or not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
    
    df = await read_file(file_path)
    max_rows = 100
    preview_df = df.head(max_rows)
    columns = preview_df.columns.tolist()
    data = preview_df.fillna("").astype(str).values.tolist()
    return {"preview_type": preview_type, "columns": columns, "data": data}

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
    summary = get_dataset_summary(df)
    suggestions = await get_ai_suggestions(summary)
    
    df_clean = df.copy()
    
    for sug_id in suggestion_ids:
        suggestion = next((s for s in suggestions if s["id"] == sug_id), None)
        if suggestion:
            df_clean = apply_suggestion(df_clean, suggestion)
    
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