# NaNja

NaNja is an intelligent web application that streamlines the entire data preprocessing workflow from messy CSV to model-ready data with speed and simplicity. Built using a FastAPI backend and a lightweight HTML/JavaScript frontend, it allows users to upload datasets, explore insightful summaries, detect common issues like nulls and duplicates, and apply basic cleaning operations with a single click. What makes NaNja truly useful is AI-powered data cleaning suggestions — such as dropping irrelevant columns, handling missing values, or recommending feature engineering all of which can also be applied instantly with one click.


## Features
- **Upload CSV**: Upload datasets via a drag-and-drop interface.
- **Dataset Summary**: View row count, column details, duplicates, and missing values.
- **Basic Cleaning**: Remove duplicates, fill missing values, and more.
- **AI Suggestions**: Get cleaning suggestions powered by Groq (requires API key), including column drops, imputations, and feature engineering ideas.
- **One-Click AI Application**: Instantly apply AI-generated suggestions to your dataset with a single click.
- **Download**: Export cleaned datasets as CSV files.


## Prerequisites
- **Python 3.10+** (for local running)
- **Docker** and **Docker Compose** (for Docker running)
- **Groq API Key** 

## Installation

### Option 1: Running Locally
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd NaNja
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python -m venv fastapiEnv
   source fastapiEnv/bin/activate  # On Windows: fastapiEnv\Scripts\activate
   ```

3. **Install Backend Dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

4. **Create `.env` File**:
   ```bash
   echo "GROQ_API_KEY=your_key_here" > .env
   ```
   Replace `your_key_here` with your Groq API key. If omitted, mock suggestions are used.

5. **Update Frontend API URLs**:
   The frontend (`script.js`, `processing.js`) uses `http://backend:8000` for API calls, which works in Docker. For local running, replace all instances of `http://backend:8000` with `http://localhost:8000`:
   - Edit `frontend/script.js` and `frontend/processing.js`.
   - Example:
     ```javascript
     // Change from:
     const response = await fetch('http://backend:8000/api/upload', ...);
     // To:
     const response = await fetch('http://localhost:8000/api/upload', ...);
     ```

6. **Run Backend**:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```

7. **Serve Frontend**:
   Use a simple HTTP server:
   ```bash
   cd frontend
   python -m http.server 8080
   ```


### Option 2: Running with Docker
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd NaNja
   ```

2. **Install Docker and Docker Compose**:
   - Install Docker: [Official Guide](https://docs.docker.com/get-docker/)
   - Install Docker Compose plugin:
     ```bash
     sudo apt update
     sudo apt install -y docker-compose-plugin
     ```

3. **Create `.env` File**:
   ```bash
   echo "GROQ_API_KEY=your_key_here" > .env
   ```
   Replace `your_key_here` with your Groq API key. If omitted, mock suggestions are used.

4. **Build and Run Containers**:
   ```bash
   docker compose up --build -d
   ```

6. **Access the App**:
   - Frontend: `http://localhost:8080`
   - Backend API: `http://localhost:8000/docs`


## Testing
1. **Prepare Test Dataset**:
   Ensure `test_datasets/Data.csv` exists. Create a sample if needed:
   ```brooke
   echo "age,salary,department\n25,50000,HR\n30,,IT\n,60000,Finance" > test_datasets/Data.csv
   ```

2. **Run Tests Locally**:
   ```bash
   source fastapiEnv/bin/activate
   pytest tests/test_main.py -v
   ```

3. **Run Tests in Docker**:
   Copy the test dataset to the backend container:
   ```bash
   docker compose cp test_datasets/Data.csv backend:/app/test_datasets/Data.csv
   ```
   Run tests:
   ```bash
   docker compose exec backend pytest /app/tests/test_main.py -v
   ```

## Usage
1. Open `http://localhost:8080`.
2. Drag and drop a CSV file or click to upload.
3. View the dataset summary and preview on the processing page.
4. Apply basic cleaning (e.g., remove duplicates) or AI-powered suggestions in just clicks.
5. Download the cleaned CSV.



## Contributing
Contributions from the community are welcomed! 
Feel free to submit issues, suggest new features, or open pull requests. Please ensure that all tests pass and follow the existing project structure and coding style.

