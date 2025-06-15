# 💬 FastAPI Chat Application

A real-time chat application built with **FastAPI** using **WebSockets**, designed to support multiple users and real-time messaging. Can be used standalone or with a React frontend.

---

## 🚀 Features

- Real-time messaging with WebSockets  
- Broadcast messages to all connected users  
- Lightweight and fast with FastAPI + Uvicorn  
- Easy to deploy with Docker or to cloud platforms

---

## 🏗️ Project Structure

├── main.py # FastAPI app with WebSocket support
├── requirements.txt # Python dependencies
├── Dockerfile # For Docker deployment
└── README.md # This file



---

## 🔧 Requirements

- Python 3.10+
- pip
- (optional) Docker

---

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/fastapi-chat-app.git
cd fastapi-chat-app


Create a virtual environment and install dependencies
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt


Run Locally
uvicorn main:app --reload


WebSocket Endpoint
ws://localhost:8000/ws

Docker
docker build -t fastapi-chat .
docker run -d -p 8000:8000 fastapi-chat

🌍 Deployment
Option 1: Deploy to Render
Create a new Web Service on Render

Use Dockerfile for deployment

Set the Start Command to:
uvicorn main:app --host 0.0.0.0 --port 10000

Pull requests are welcome. For major changes, please open an issue first to discuss what you’d like to change.
