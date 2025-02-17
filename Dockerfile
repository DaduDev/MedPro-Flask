FROM python:3.9-slim-buster

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt  # This installs your other deps

# Install Gunicorn!  This is the missing piece
RUN pip install gunicorn

COPY . .

EXPOSE 8080  # This is good to keep

CMD gunicorn --bind :8080 --workers 1 --threads 1 app:app