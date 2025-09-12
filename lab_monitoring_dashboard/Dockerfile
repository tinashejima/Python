FROM python:3.10-slim

WORKDIR /app

# Install system dependencies for pyhive 
RUN apt-get update && apt-get install -y gcc libsasl2-dev

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app code
COPY . .

EXPOSE 5000

CMD ["python", "app.py"]