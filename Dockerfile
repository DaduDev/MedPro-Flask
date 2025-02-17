FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install the necessary system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1

COPY . .

EXPOSE 8080

CMD gunicorn --bind :8080 --workers 1 --threads 1 app:app