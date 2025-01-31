# Stage 1: Install heavy dependencies first (for caching)
FROM python:3.11-slim AS dependencies
WORKDIR /app

# Install system-level dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libstdc++6 \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (leverage Docker caching)
COPY requirements.txt .

# Install heavy Python dependencies separately
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch transformers

# Stage 2: Final application image
FROM python:3.11-slim AS final
WORKDIR /app

# Copy preinstalled dependencies from the first stage
COPY --from=dependencies /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy the rest of the application files
COPY . .

# Expose the port the application will run on
EXPOSE 8000

# Default command to run the application
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
