FROM python:3.9-slim
ENV PYTHONUNBUFFERED True

WORKDIR /root
COPY /app /root/app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8080
CMD uvicorn app.main:app --port=8080 --host=0.0.0.0
