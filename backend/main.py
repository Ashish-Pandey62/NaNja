import os
import uuid
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
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

SUGGESTIONS_STORE = {}

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
    null_percentages = (df.isnull().sum() / len(df) * 100).round(2).astype(str).to_dict()
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
        "columns_list": df.columns.tolist()
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
Suggestions may include: dropping columns with high nulls, filling nulls, feature engineering (e.g., binning), encoding categorical variables, or normalizing numerical columns.

Rules:
- Only suggest operations on existing columns: {summary["columns_list"]}.
- For dropping columns, specify the exact column name.
- For filling nulls, suggest a specific method (mean, median, mode).
- For feature engineering, suggest clear transformations (e.g., binning, one-hot encoding).
- Ensure suggestions are actionable and specific to the dataset.

Return the suggestions as a JSON object in the format:
{{"suggestions": [{{"id": 1, "text": "Drop column X due to high nulls", "type": "drop", "column": "X"}}, ...]}}
Each suggestion must have:
- 'id' (integer, starting from 1)
- 'text' (clear description)
- 'type' ('drop', 'fill', 'feature', or 'transform')
- 'column' (target column name, if applicable)

Return ONLY the JSON object, no explanations or markdown.
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
        
        cleaned_response = response.strip()
        if cleaned_response.startswith("```json"):
            cleaned_response = cleaned_response[len("```json"):].strip()
        if cleaned_response.startswith("```"):
            cleaned_response = cleaned_response[len("```"):].strip()
        if cleaned_response.endswith("```"):
            cleaned_response = cleaned_response[:-len("```")].strip()
        logger.info(f"Cleaned LLM response: {cleaned_response}")
        
        try:
            suggestions = json.loads(cleaned_response).get("suggestions", [])
            logger.info(f"Parsed JSON suggestions: {suggestions}")
            if not suggestions:
                logger.warning("No suggestions found in JSON response")
                return [{"id": 1, "text": "No suggestions provided by AI", "type": "drop"}]
            return suggestions
        except json.JSONDecodeError as json_err:
            logger.warning(f"JSON parsing failed: {str(json_err)}")
            suggestions = []
            lines = cleaned_response.split('\n')
            for i, line in enumerate(lines, 1):
                line = line.strip()
                if line.startswith('- ') or line.startswith('* '):
                    text = line[2:].strip()
                    suggestion_type = "drop"
                    column = None
                    if "drop" in text.lower():
                        suggestion_type = "drop"
                        column_match = re.search(r"column\s+([a-zA-Z_][a-zA-Z0-9_]*)", text, re.IGNORECASE)
                        column = column_match.group(1) if column_match else None
                    elif "fill" in text.lower():
                        suggestion_type = "fill"
                        column_match = re.search(r"column\s+([a-zA-Z_][a-zA-Z0-9_]*)", text, re.IGNORECASE)
                        column = column_match.group(1) if column_match else None
                    elif "bins" in text.lower() or "feature" in text.lower():
                        suggestion_type = "feature"
                        column_match = re.search(r"for\s+([a-zA-Z_][a-zA-Z0-9_]*)", text, re.IGNORECASE)
                        column = column_match.group(1) if column_match else None
                    elif "normalize" in text.lower() or "encode" in text.lower():
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

async def apply_suggestion(df: pd.DataFrame, suggestion: Dict) -> pd.DataFrame:
    df_clean = df.copy()
    suggestion_type = suggestion.get("type")
    column = suggestion.get("column")
    text = suggestion.get("text", "").lower()

    try:
        if suggestion_type == "drop" and column:
            if column not in df_clean.columns:
                raise ValueError(f"Column {column} does not exist")
            df_clean = df_clean.drop(columns=[column])
            logger.info(f"Dropped column {column}")
        
        elif suggestion_type == "fill" and column:
            if column not in df_clean.columns:
                raise ValueError(f"Column {column} does not exist")
            if "mean" in text and pd.api.types.is_numeric_dtype(df_clean[column]):
                df_clean[column] = df_clean[column].fillna(df_clean[column].mean())
            elif "median" in text and pd.api.types.is_numeric_dtype(df_clean[column]):
                df_clean[column] = df_clean[column].fillna(df_clean[column].median())
            elif "mode" in text:
                df_clean[column] = df_clean[column].fillna(df_clean[column].mode()[0])
            else:
                raise ValueError(f"Invalid fill method for column {column}")
            logger.info(f"Filled nulls in column {column}")
        
        elif suggestion_type == "feature" and column:
            if column not in df_clean.columns:
                raise ValueError(f"Column {column} does not exist")
            if "bin" in text and pd.api.types.is_numeric_dtype(df_clean[column]):
                new_col = f"{column}_binned"
                df_clean[new_col] = pd.qcut(df_clean[column], q=5, duplicates="drop")
                logger.info(f"Created binned column {new_col}")
            elif "one-hot" in text:
                if df_clean[column].nunique() > 10:
                    raise ValueError(f"Column {column} has too many unique values for one-hot encoding")
                encoded = pd.get_dummies(df_clean[column], prefix=column)
                df_clean = pd.concat([df_clean.drop(columns=[column]), encoded], axis=1)
                logger.info(f"One-hot encoded column {column}")
            else:
                raise ValueError(f"Unsupported feature engineering for column {column}")
        
        elif suggestion_type == "transform" and column:
            if column not in df_clean.columns:
                raise ValueError(f"Column {column} does not exist")
            if "normalize" in text and pd.api.types.is_numeric_dtype(df_clean[column]):
                df_clean[column] = (df_clean[column] - df_clean[column].min()) / (df_clean[column].max() - df_clean[column].min())
                logger.info(f"Normalized column {column}")
            elif "encode" in text:
                df_clean[column] = df_clean[column].astype("category").cat.codes
                logger.info(f"Label encoded column {column}")
            else:
                raise ValueError(f"Unsupported transformation for column {column}")
        
        else:
            # Fallback to LLM for complex suggestions
            if not llm_client:
                raise ValueError("LLM service not available for complex suggestion")
            
            prompt = f"""
I have a pandas DataFrame with these columns: {df.columns.tolist()}
The suggested preprocessing step is: "{suggestion['text']}"

Generate a Python code snippet that:
1. Takes a DataFrame 'df' as input
2. Applies the suggested transformation
3. Returns the modified DataFrame as 'df_clean'

The code should:
- Be a single expression or small block (1-3 lines)
- Only use pandas operations
- Handle edge cases (nulls, wrong types)
- Not modify the original DataFrame
- Only operate on existing columns: {df.columns.tolist()}

Return ONLY the Python code, no explanations or markdown.

Example for "Drop column X":
df_clean = df.drop(columns=['X'])

Example for "Fill nulls with median":
df_clean = df.fillna(df.median())

Now generate code for: "{suggestion['text']}"
"""
            logger.info(f"Generating LLM code for suggestion: {suggestion['text']}")
            chat_completion = llm_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=500,
            )
            generated_code = chat_completion.choices[0].message.content.strip()
            if generated_code.startswith("```python"):
                generated_code = generated_code[len("```python"):].strip()
            if generated_code.startswith("```"):
                generated_code = generated_code[len("```"):].strip()
            if generated_code.endswith("```"):
                generated_code = generated_code[:-len("```")].strip()
            
            # Print the generated code to the terminal
            print(f"\n=== Generated Code for Suggestion: {suggestion['text']} ===\n{generated_code}\n====================================\n")
            logger.info(f"Generated code for suggestion '{suggestion['text']}':\n{generated_code}")
            
            safe_globals = {
                'pd': pd,
                'df': df_clean,
                '__builtins__': {
                    'list': list,
                    'dict': dict,
                    'str': str,
                    'int': int,
                    'float': float,
                    'bool': bool,
                    'range': range,
                }
            }
            safe_locals = {'df_clean': None}
            try:
                exec(generated_code, safe_globals, safe_locals)
            except Exception as e:
                raise ValueError(f"Failed to execute generated code: {str(e)}")
            
            if 'df_clean' not in safe_locals or safe_locals['df_clean'] is None:
                raise ValueError("Generated code did not produce a valid DataFrame")
            
            df_clean = safe_locals['df_clean']
            logger.info(f"Applied LLM-generated suggestion: {generated_code}")
        
        return df_clean
    
    except Exception as e:
        logger.error(f"Error processing suggestion '{suggestion['text']}': {str(e)}")
        raise ValueError(f"Failed to apply suggestion: {str(e)}")

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
        raise HTTPException(status_code=404, detail=f"File not found for file_id: {file_id}")
    
    df = await read_file(file_path)
    summary = get_dataset_summary(df)
    return summary

@app.get("/api/preview/{file_id}", response_model=DatasetPreview)
async def get_preview(file_id: str, cleaned_file_id: Optional[str] = None):
    if cleaned_file_id:
        file_path = CLEANED_DIR / f"{

cleaned_file_id}.csv"
        preview_type = "cleaned"
        if not file_path.exists():
            logger.error(f"Cleaned file not found at {file_path}")
            raise HTTPException(status_code=404, detail=f"Cleaned file not found for cleaned_file_id: {cleaned_file_id}")
    else:
        file_path = next(UPLOAD_DIR.glob(f"{file_id}.*"), None)
        preview_type = "original"
        if not file_path or not file_path.exists():
            logger.error(f"Original file not found at {file_path}")
            raise HTTPException(status_code=404, detail=f"File not found for file_id: {file_id}")
    
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
        raise HTTPException(status_code=404, detail=f"File not found for file_id: {file_id}")
    
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
        raise HTTPException(status_code=404, detail=f"File not found for file_id: {file_id}")
    
    df = await read_file(file_path)
    summary = get_dataset_summary(df)
    suggestions = await get_ai_suggestions(summary)
    
    # Store suggestions with proper IDs
    suggestions_with_ids = []
    for i, sug in enumerate(suggestions, 1):
        suggestions_with_ids.append({
            "id": i,
            "text": sug.get("text", "No description"),
            "type": sug.get("type", "transform"),
            "column": sug.get("column")
        })
    
    SUGGESTIONS_STORE[file_id] = suggestions_with_ids
    logger.info(f"Stored suggestions for file_id {file_id}: {suggestions_with_ids}")
    return {"suggestions": suggestions_with_ids}

@app.post("/api/apply-suggestions/{file_id}", response_model=Dict)
async def apply_suggestions(file_id: str, suggestion_ids: List[int]):
    logger.info(f"Received apply-suggestions request for file_id: {file_id}, suggestion_ids: {suggestion_ids}")
    file_path = next(UPLOAD_DIR.glob(f"{file_id}.*"), None)
    if not file_path or not file_path.exists():
        logger.error(f"File not found for file_id: {file_id}")
        raise HTTPException(status_code=404, detail=f"File not found for file_id: {file_id}")
    
    # Get stored suggestions
    suggestions = SUGGESTIONS_STORE.get(file_id, [])
    logger.info(f"Stored suggestions for file_id {file_id}: {suggestions}")
    if not suggestions:
        logger.error("No suggestions available for file_id: {file_id}")
        raise HTTPException(status_code=400, detail="No suggestions available. Generate suggestions first.")
    
    df = await read_file(file_path)
    df_clean = df.copy()
    applied_suggestions = []
    failed_suggestions = []
    
    # Validate suggestion IDs
    valid_ids = [s["id"] for s in suggestions]
    invalid_ids = [sid for sid in suggestion_ids if sid not in valid_ids]
    if invalid_ids:
        logger.warning(f"Invalid suggestion IDs: {invalid_ids}")
        failed_suggestions.extend([f"Suggestion ID {sid}: Not found" for sid in invalid_ids])
    
    # Apply selected suggestions
    for sug_id in [sid for sid in suggestion_ids if sid in valid_ids]:
        suggestion = next((s for s in suggestions if s["id"] == sug_id), None)
        if not suggestion:
            logger.warning(f"Suggestion ID {sug_id} not found")
            failed_suggestions.append(f"Suggestion ID {sug_id}: Not found")
            continue
        try:
            logger.info(f"Applying suggestion {sug_id}: {suggestion['text']}")
            df_clean = await apply_suggestion(df_clean, suggestion)
            applied_suggestions.append(suggestion["text"])
        except Exception as e:
            logger.error(f"Error applying suggestion {sug_id}: {str(e)}")
            failed_suggestions.append(f"Suggestion '{suggestion['text']}': {str(e)}")
            continue
    
    if not applied_suggestions:
        logger.error(f"No suggestions applied. Failed suggestions: {failed_suggestions}")
        raise HTTPException(
            status_code=400,
            detail=f"No suggestions were applied. Errors: {'; '.join(failed_suggestions)}"
        )
    
    # Save cleaned data
    cleaned_file_id = str(uuid.uuid4())
    cleaned_path = CLEANED_DIR / f"{cleaned_file_id}.csv"
    df_clean.to_csv(cleaned_path, index=False)
    
    # Prepare response with feedback
    response = {
        "cleaned_file_id": cleaned_file_id,
        "applied_suggestions": applied_suggestions,
        "failed_suggestions": failed_suggestions
    }
    logger.info(f"Applied suggestions: {applied_suggestions}")
    if failed_suggestions:
        logger.warning(f"Failed suggestions: {failed_suggestions}")
    print(f"\n=== Apply Suggestions Result for file_id: {file_id} ===\n"
          f"Applied: {applied_suggestions}\n"
          f"Failed: {failed_suggestions}\n"
          f"Cleaned File ID: {cleaned_file_id}\n"
          f"====================================\n")
    return response

@app.get("/api/download/{cleaned_file_id}")
async def download_file(cleaned_file_id: str):
    file_path = CLEANED_DIR / f"{cleaned_file_id}.csv"
    if not file_path.exists():
        logger.error(f"Download file not found at {file_path}")
        raise HTTPException(status_code=404, detail=f"Cleaned file not found for cleaned_file_id: {cleaned_file_id}")
    
    return FileResponse(file_path, filename="cleaned_dataset.csv", media_type="text/csv")