FROM python:3.12

WORKDIR /app/backend

COPY . /app/backend
RUN pip install -r src/requirements.txt

WORKDIR /app/backend/src

CMD ["uvicorn", "main:app","--host", "0.0.0.0", "--port", "10000"]
