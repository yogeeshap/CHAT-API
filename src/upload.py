from fastapi import FastAPI, WebSocket
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, storage, firestore
import uuid
import io

app = FastAPI()

# Initialize Firebase
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'your-bucket-name.appspot.com'
})
firestore_client = firestore.client()


@app.websocket("/ws/record")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    file_id = str(uuid.uuid4())
    file_name = f"recordings/{file_id}.webm"
    binary_stream = io.BytesIO()

    user_id = None  # You can extract from session/cookie if needed

    try:
        while True:
            data = await websocket.receive_bytes()
            binary_stream.write(data)
    except Exception as e:
        print("WebSocket closed:", e)
    finally:
        # Upload to Firebase Storage
        binary_stream.seek(0)
        bucket = storage.bucket()
        blob = bucket.blob(file_name)
        blob.upload_from_file(binary_stream, content_type="video/webm")

        # (Optional) Make the file public
        blob.make_public()

        # Store metadata in Firestore
        metadata = {
            "user_id": user_id or "anonymous",
            "filename": file_name,
            "public_url": blob.public_url,
            "uploaded_at": datetime.utcnow(),
            "content_type": "video/webm"
        }

        await firestore_client.collection("Recordings").add(metadata)
        print("Metadata saved to Firestore.")
