# Use a lightweight base image
FROM python:3.11-slim

# Set a working directory inside the container
WORKDIR /app

# Install necessary system dependencies (if required)
RUN apt-get update && apt-get install -y --no-install-recommends gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (to leverage Docker layer caching for dependencies)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY . .

# Expose the port the application will run on
EXPOSE 8000

# Default command to run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
