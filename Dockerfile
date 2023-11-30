FROM python:3.9-slim
ENV PYTHONUNBUFFERED True

WORKDIR /workspace
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
    
COPY . .

EXPOSE 8080
CMD uvicorn app.main:app --port=8080 --host=0.0.0.0
