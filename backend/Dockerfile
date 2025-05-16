# Use official Python slim image for smaller size
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy backend files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for uploads and cleaned files
RUN mkdir -p uploads cleaned
RUN chmod -R 777 uploads cleaned

# Expose port
EXPOSE 8000

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]