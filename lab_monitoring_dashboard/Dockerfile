# Using Python 3.9 slim image as base
FROM python:3.9-slim

# Setting up working directory
WORKDIR /app

# Installing system dependencies required for PyHive and SASL
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libsasl2-dev \
    libsasl2-modules \
    libsasl2-modules-gssapi-mit \
    libkrb5-dev \
    libssl-dev \
    libffi-dev \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Installing Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copying application code
COPY app.py /app/

# Creating templates directory and copying HTML files
RUN mkdir -p templates
COPY templates/ templates/

# Exposing port
EXPOSE 5000

# Setting environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production

# Creating a non-root user for security
RUN useradd -r -s /bin/false flaskuser && \
    chown -R flaskuser:flaskuser /app
USER flaskuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "4", "--timeout", "120", "app:app"]