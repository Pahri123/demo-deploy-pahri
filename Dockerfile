FROM python:3.9-slim
ENV PYTHONUNBUFFERED True

WORKDIR /root
COPY /app /root/app

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /root/app/requirements.txt

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
