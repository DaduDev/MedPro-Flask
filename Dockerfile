FROM python:3.9-slim-buster

WORKDIR /app

# Copy the requirements file FIRST
COPY requirements.txt .

# Install the dependencies, including gunicorn
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port (important for Render)
EXPOSE 8080

# Use CMD to start Gunicorn.  No need for a separate Procfile
CMD gunicorn --bind :8080 --workers 1 --threads 1 app:app # Replace app:app with your app's entry point