FROM python:3.10-slim

WORKDIR /app

# Upgrade pip to ensure the latest package wheel parsing rules apply
RUN pip install --no-cache-dir --upgrade pip

# Force explicit installation of core dependencies straight into the image layers
RUN pip install --no-cache-dir langchain-text-splitters streamlit

# Install CPU-only PyTorch to prevent massive CUDA downloads and network timeouts
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy requirements and handle remaining downstream libraries
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.fileWatcherType", "none"]
