import asyncio
import base64
from collections import defaultdict
import json
import threading
import os
import uuid
from auth import create_secure_cookie, decode_secure_cookie, generate_otp, parse_cookie_header
import firebase_admin
from datetime import datetime
from fastapi import FastAPI,Query,Path,Body, WebSocket, WebSocketDisconnect,HTTPException,Request, Response, Form, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from typing import Optional, Set, Dict, List,DefaultDict
from firebase_admin import credentials, firestore_async,firestore
from fastapi.responses import RedirectResponse, JSONResponse
from schema.chat_schema import (AddUserToRoom, RoomUser,
                                # RoomMember,
                                RoomUserMap,
                                UserModel,
                                VerifyOtpRequest,
                                RoomUpdatePayload
                                )
from cachetools import TTLCache

# Initialize Firebase
cred = credentials.Certificate("firebase_key.json")
firebase_admin.initialize_app(cred)
SESSION_COOKIE_NAME = "super-secret-key"

print(os.path.abspath("firebase_key.json"),'pathhh')
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath("firebase_key.json")
# Initialize Firestore DB
sync_db = firestore.client()
async_db = firestore_async.AsyncClient()


app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://localhost:5173","https://localhost:3000"],  # Replace * with React frontend origin in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store connected clients
clients = []

clients = {}  # client_id: websocket
clients_lock = asyncio.Lock()
rooms_lock = asyncio.Lock()
rooms = {}
listeners: Dict[str, threading.Thread] = {}
stop_flags: Dict[str, threading.Event] = {}
room_hosts = {}
hosts: dict[str, str] = {}  # room_id -> host_user_id
# room_id -> list of active WebSocket connections
room_connections: Dict[str, List[WebSocket]] = DefaultDict(list)
ws_queues: dict[str,WebSocket, asyncio.Queue] = {}
otp_cache = TTLCache(maxsize=1000, ttl=300)
# message_queues = defaultdict(dict)

connections = defaultdict(set)  # room_id -> set of websockets
ws_queues = defaultdict(dict)   # room_id -> { websocket: queue }
user_ids = defaultdict(dict)    # room_id -> { websocket: user_id }
main_event_loop = None

@app.on_event("startup")
async def on_startup():
    global main_event_loop
    main_event_loop = asyncio.get_event_loop()


async def fetch_user_details(member_refs: list[dict]):
    users = []
    for item in member_refs:
        member_ref = item["memberId"]
        room_ref = item["roomId"]
        member_doc = await member_ref.get()
        room_doc = await room_ref.get()
        if member_doc.exists:
            user_data = member_doc.to_dict()
            room_data = room_doc.to_dict()
            users.append({
                "user_id": member_doc.id,
                "username": user_data.get("username"),
                "email": user_data.get("email"),
                "room_id":room_doc.id,
                "room_name":room_data['name']
            })
    return users

async def get_user_details(user_room_map:List[RoomUserMap]):

    ref_list = []

    for urm in user_room_map:
        user_ref = None
        # Determine user reference based on available info
        if urm.username:
            user_query = async_db.collection("User").where("username", "==", urm.username)
            async for doc in user_query.stream():
                user_ref = doc.reference
                break
        elif urm.email:
            user_query = async_db.collection("User").where("email", "==", urm.email)
            async for doc in user_query.stream():
                user_ref = doc.reference
                break
        elif urm.phone_number:
            user_query = async_db.collection("User").where("phone_number", "==", urm.phone_number)
            async for doc in user_query.stream():
                user_ref = doc.reference
                break

        # Get RoomMember docs
        elif urm.room_id:
            room_ref = async_db.collection("Room").document(urm.room_id)
            if urm.limit:
                rm_query = async_db.collection("RoomMember").where("roomId", "==", room_ref).limit(10)
            else:
                rm_query = async_db.collection("RoomMember").where("roomId", "==", room_ref)
            async for rm_doc in rm_query.stream():
                ref_list.append(rm_doc.to_dict())

        elif urm.user_id:
            user_ref = async_db.collection("User").document(urm.user_id)
            if urm.limit:
                rm_query = async_db.collection("RoomMember").where("memberId", "==", user_ref).limit(10)
            else:
                rm_query = async_db.collection("RoomMember").where("memberId", "==", user_ref)
            async for rm_doc in rm_query.stream():
                ref_list.append(rm_doc.to_dict())

        else:
            if urm.limit:
                room_ref = async_db.collection("RoomMember").limit(10)
            else:
                room_ref = async_db.collection("RoomMember")

            async for rm_doc in room_ref.stream():
                ref_list.append(rm_doc.to_dict())

    results = await fetch_user_details(ref_list)

    return results
        


def generate_search_index(*args: str) -> list[str]:
    index = set()
    for arg in args:
        if not arg:
            continue
        arg = arg.lower()
        for i in range(1, len(arg) + 1):
            index.add(arg[:i])
    return list(index)


@app.post("/login")
async def login(response: Response, user:UserModel):

    user_data = None
    user_id = None

    user_query = async_db.collection("User").where("email", "==", user.email)
    async for doc in user_query.stream():
        user_data = doc.to_dict()
        user_id = doc.id

    if not user_data:
        raise HTTPException(status_code=404, detail="User Not exists")

    if user_data and user_data["password"] != user.password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    session_data = {"user_id": user_id}
    cookie = create_secure_cookie(session_data)

    response = JSONResponse(content={"message": "Login successful","user":{
                                                        "username":user_data['username'],
                                                        "email":user_data['email'],
                                                        # "profile_pic":user_data['profile_pic']
                                                        }
                                                        })

    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=cookie,
        httponly=True,
        secure=True,  # Set to True in production (HTTPS)
        samesite="None",
        max_age=3600  # 1 hour
    )

    return response
    
 
# Dependency to get current user
def get_current_session(request: Request):
    cookie = request.cookies.get(SESSION_COOKIE_NAME)
    if not cookie:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    session = decode_secure_cookie(cookie)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    
    return session

# Admin-only route
def require_admin(session: dict = Depends(get_current_session)):
    if session["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admins only")
    return session


# Logout route
@app.get("/logout")
async def logout(response: Response):
    response = RedirectResponse(url="/", status_code=200)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return response


@app.post("/send_otp")
def send_otp(user: UserModel):
    
    user_obj = {
                "phone_number":user.phone_number,
 
                }
    users = get_user_details(RoomUserMap(**user_obj))
    
    if not users:
        raise HTTPException(status_code=404, detail="Phone number not registered")

    otp = generate_otp()
    otp_cache[user] = otp

    # Simulate SMS
    print(f"Sending OTP to {user.phone_number}: {otp}")

    return {"message": "OTP sent"}


@app.post("/verify_otp")
def verify_otp(data: VerifyOtpRequest):
    stored_otp = otp_cache.get(data.phone)

    if not stored_otp:
        raise HTTPException(status_code=400, detail="No OTP found or it expired")

    if data.otp != stored_otp:
        raise HTTPException(status_code=400, detail="Incorrect OTP")

    del otp_cache[data.phone]  # Remove used OTP
    return {"message": "OTP verified", "user": users[data.phone]}


@app.get("/room_users",dependencies = [Depends(get_current_session)])
async def room_users(
    room_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    limit: Optional[str] = Query(None),
    session:dict = Depends(get_current_session)
    ):

    user_obj = {'user_id':session['user_id'],'limit':limit}

    if user_id:
        user_obj.update({'user_id':user_id})

    if room_id:
        user_obj.update({'room_id':room_id})

    users = await get_user_details([RoomUserMap(**user_obj)])
    count = len(users)
    return {"users": users,'count':count}


@app.get("/room",dependencies = [Depends(get_current_session)])
async def get_room(
    room_id: str = Query(..., description="Room Id")
    ):

    room_data = {}
    if room_id:
        room_ref = async_db.collection("Room").document(room_id)
        doc = await room_ref.get()
        if doc.exists:
            room_data = doc.to_dict()
            room_data.update({'room_id':doc.id})

    return {"room": room_data}

@app.patch("/room/{room_id}",dependencies = [Depends(get_current_session)])
async def update_room(
    room_id: str = Path(..., description="Room Id"),
    payload: RoomUpdatePayload = Body(...)
    ):

    try:
        if room_id:
            room_ref = async_db.collection("Room").document(room_id)
            await room_ref.update({
                            "name": payload.name
                        })

            return {"message":"Successfully Updated"}
    except:
        return JSONResponse(content={"message": "Failed"})
        


@app.get("/search_users/",dependencies = [Depends(get_current_session)])
async def search_users(q: str = Query(..., description="Search query"),):
    users_list = []
    user_query = async_db.collection("User").where("search_index", "array_contains", q)
    async for rm_doc in user_query.stream():   
        users_list.append({
                            'username':rm_doc.to_dict().get('username'),
                            'user_id':rm_doc.id
                            })

    return {"users": users_list}

@app.get("/other_user_details")
async def user_details(response: Response,
                    user: UserModel,
                    session = [Depends(get_current_session)]):

    users = await get_user_details([user])

    return {"users": users}

async def add_user(user_obj: UserModel):
    # Check if user already exists
    user_query = async_db.collection("User").where("email", "==", user_obj.email)
    async for doc in user_query.stream():
        raise HTTPException(
            status_code=409,
            detail={
                "error": "User already exists",
                "code": "user_already_exists",
                "user_id": doc.id
            }
        )

    # Create new user
    new_user_ref = async_db.collection("User").document()
    search_index = generate_search_index(user_obj.username, user_obj.email)
    await new_user_ref.set({
        "password": user_obj.password,
        "email": user_obj.email,
        "username": user_obj.username,
        'search_index':search_index,
        'public_id':str(uuid.uuid4())
        
    })

    return {"message": "User added", "id": new_user_ref.id}


async def add_mbr_to_room(room_member:RoomUserMap):
  
    user_ref = async_db.collection("User").document(room_member.user_id)

    room_member_id = f"{room_member.room_id}_{room_member.user_id}"
    room_member_ref = async_db.collection("RoomMember").document(room_member_id)

    await room_member_ref.set(
        {
            "roomId":room_member.room_ref,
            "memberId":user_ref
            })
    
    room_ref = await room_member.room_ref.get()
    room_doc = room_ref.to_dict()
    user_doc = await user_ref.get()
    user_data = user_doc.to_dict()

    return {"message": "Room created and members added",
             "room_user": {
                 'room_id':room_member_ref.id,
                 'user_id':user_ref.id,
                 'username':user_data.get('username'),
                 'email':user_data.get('email'),
                 'room_name':room_doc.get('name')
             }
             }


@app.post("/register")
async def user_details(response: Response,user:UserModel):
    users = await add_user(user)
    response.status_code =  201
    return users

@app.patch("/add_user_to_room/{room_id}")
async def add_user_to_room(
    room_id: str = Path(..., description="Room Id"),
    room_user_map: AddUserToRoom = Body(...)
    ,session:dict = Depends(get_current_session)
    ):
    try:
        room_ref = async_db.collection("Room").document(room_id)
        room_rd = await room_ref.get()
        if not room_rd.exists:
            raise HTTPException(status_code=401, detail="Room Does not exists")

        room_user_res = []
        for usr in room_user_map.users:
            room_obj = {
                "room_id":room_ref.id,
                "user_id":usr,
                "room_ref":room_ref
            }
            room_res = await add_mbr_to_room(RoomUserMap(**room_obj))
            room_user_dtl = room_res['room_user']
            room_user_res.append(room_user_dtl)
        return {"message": f"Users added to Room", "room_user": room_user_res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/create_room/")
async def create_room(room_payload:RoomUser,session:dict = Depends(get_current_session)):
    try:

        room_data = None
        room_name = room_payload.room_name
        users = room_payload.users

        room_filter_ref = async_db.collection("Room").where("name", "==", room_name)

        async for doc in room_filter_ref.stream():
            room_data = doc.to_dict()


        if room_data:
            raise HTTPException(status_code=409, detail="Room already exists")
        
        room_ref = async_db.collection("Room").document()
        await room_ref.set({ "name":room_name})

        users.append(session['user_id'])

        
        for usr in users:
            room_obj = {
                "room_id":room_ref.id,
                "user_id":usr,
                "room_ref":room_ref
            }
       
            room_res = await add_mbr_to_room(RoomUserMap(**room_obj))

        room_res = {
                    "room_id":room_ref.id,
                    "user_id":session['user_id'],
                    "room_name":room_name
                    }

        return {"message": "Room Created", "room": room_res}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def sender_loop(websocket: WebSocket, room_id: str):
    queue = ws_queues[room_id].setdefault(websocket, asyncio.Queue(maxsize=100))
    try:
        while True:
            message = await queue.get()
            await websocket.send_text(message)
    except Exception as e:
        print(f"Sender loop error: {e}")
    finally:
        print("Sender loop exiting")
        ws_queues[room_id].pop(websocket, None)
        connections[room_id].discard(websocket)
        user_ids[room_id].pop(websocket, None)


def firestore_listener(room_id: str):
    stop_flag = stop_flags[room_id]

    def on_snapshot(col_snapshot, changes, read_time):
        for change in changes:
            if stop_flag.is_set():
                return

            if change.type.name in ('ADDED', 'MODIFIED'):
                doc = change.document.to_dict()
                doc["id"] = change.document.id
                doc["timestamp"] = doc.get("timestamp").isoformat() if doc.get("timestamp") else None
                if doc.get("edited_at"):
                    doc["edited_at"] = doc["edited_at"].isoformat()

                message = json.dumps({
                    "type": "message_update",
                    "room_id": room_id,
                    "message": doc,
                })

                async def enqueue():
                    for ws, queue in ws_queues[room_id].copy().items():
                        try:
                            queue.put_nowait(message)
                        except asyncio.QueueFull:
                            print(f"[Queue Full] Skipping message for {ws.client}")

                if main_event_loop and not main_event_loop.is_closed():
                    asyncio.run_coroutine_threadsafe(enqueue(), main_event_loop)

    print(f"Starting Firestore listener for room: {room_id}")
    ref = sync_db.collection("Post").where("room_id", "==", room_id)
    listener = ref.on_snapshot(on_snapshot)

    stop_flag.wait()
    listener.unsubscribe()


@app.websocket("/ws/chat/{room_id}/{user_id}")
async def websocket_endpoint(websocket: WebSocket, room_id: str, user_id: str):
    await websocket.accept()

    if room_id not in stop_flags:
        stop_flags[room_id] = threading.Event()
        threading.Thread(target=firestore_listener, args=(room_id,), daemon=True).start()

    connections[room_id].add(websocket)
    ws_queues[room_id][websocket] = asyncio.Queue(maxsize=100)

    # optional: assign user_id here
    user_ids[room_id][websocket] = user_id

    sender = asyncio.create_task(sender_loop(websocket, room_id))

    async def get_initial():
        messages_ref = async_db.collection("Post").where("room_id", "==", room_id)
        docs = messages_ref.order_by("timestamp", direction=firestore_async.Query.DESCENDING).limit(50).stream()

        messages = []
        async for doc in docs:
            doc_dict = doc.to_dict()
            messages.append({
                **doc_dict,
                "id": doc.id,
                "timestamp": doc_dict["timestamp"].isoformat(),
                "edited_at": doc_dict.get("edited_at") and doc_dict["edited_at"].isoformat()
            })

        return messages


    async def get_older(start_timestamp):
        ts = datetime.fromisoformat(start_timestamp)
        messages_ref = async_db.collection("Post").where("room_id", "==", room_id)
        docs = messages_ref.order_by("timestamp").end_before({"timestamp": ts}).limit_to_last(50).stream()
        return [doc.to_dict() | {
                    "id": doc.id,
                    "timestamp": doc.to_dict()["timestamp"].isoformat(),
                    "edited_at": doc.to_dict().get("edited_at") and doc.to_dict()["edited_at"].isoformat()
                } for doc in docs]

    # Send initial history
    history = await get_initial()
    await websocket.send_json({"type": "history", "messages": history})


    try:
        while True:
            data = await websocket.receive_text()
            data_json = json.loads(data)
     
            if data_json["type"] == "get_initial":
                history = await get_initial()
                await websocket.send_json({"type": "initial", "messages": history})

            elif data_json["type"] == "load_older":
                start_ts = data_json.get("start_timestamp")
                if start_ts:
                    older = await get_older(start_ts)
                    await websocket.send_json({"type": "older_messages", "messages": older})

            if data_json["type"] == "message":

                content = data_json.get("content", "")
                filename = data_json.get("filename",'')
                content_type = data_json.get("content_type",'')
                b64_data = data_json.get("data",'')
                image_height = data_json.get("image_height",'')
                image_width = data_json.get("image_width",'')

                msg_ref = async_db.collection("Post").document()
                await msg_ref.set({
                    "author_id": user_id,
                    "content": content,
                    "timestamp": datetime.utcnow(),
                    "edited_at": None,
                    "reactions": {},
                    "room_id":room_id,
                    "filename":filename,
                    "content_type":content_type,
                    "f_data":b64_data,
                    "image_height":image_height,
                    "image_width":image_width
                })
               
            elif data_json["type"] == "edit":
                msg_id = data_json.get("id")
                new_content = data_json.get("new_content")

                if msg_id and new_content:
                    msg_ref = async_db.collection("Post").document(msg_id)
                    await msg_ref.update({
                        "content": new_content,
                        "edited_at": datetime.utcnow()
                    })
          

            elif data_json["type"] == "reaction":
                msg_id = data_json.get("id")
                reaction = data_json.get("reaction")

                if msg_id and reaction:

                    msg_ref = async_db.collection("Post").document(msg_id)
                    doc = await msg_ref.get()
                    if not doc.exists:
                        return
                    data = doc.to_dict()
                    reactions = data.get("reactions", {})
                    users = reactions.get(reaction, [])
                    if user_id not in users:
                        users.append(user_id)
                    reactions[reaction] = users
                    await msg_ref.update({"reactions": reactions})
        
            elif data_json["type"] == "typing":
                is_typing = data_json.get("is_typing", False)
                # Broadcast typing directly to clients (not via Firestore)
                message = {
                    "type": "typing",
                    "user_id": user_id,
                    "is_typing": is_typing
                }
                dead = []
                for ws in connections.get(room_id, set()).copy():
                    try:
                        await ws.send_json(message)
                    except:
                        dead.append(ws)
                for d in dead:
                    connections[room_id].pop(d)

            elif  data_json["type"] == "offer" or data_json["type"] == "answer" or data_json["type"] == "candidate":
                data_json["sender"] = user_id
                # Forward signaling message to target user
                for ws in connections.get(room_id, set()).copy():
                    try:
                        await ws.send_json(data_json)
                    except:
                        connections[room_id].pop(ws)


            elif data_json["type"] == "join_call":

                connections.setdefault(room_id, set()).add(websocket)
                
                if room_id not in hosts:
                    hosts[room_id] = user_id
                    is_host = True
                else:
                    is_host = False

                message = {
                    "type": "join_call",
                    "sender": user_id,
                    "is_host": is_host,
                    "host": hosts[room_id],
                    "user_count": len(connections.get(room_id, [])),
                    "users": list(user_ids[room_id].values()),
                }
        
                for ws in connections.get(room_id, set()).copy():
                    try:
                        if ws != websocket:
                            await ws.send_json(message)
                    except:
                        connections[room_id].pop(ws)
                        user_ids[room_id].pop(ws, None)

    except WebSocketDisconnect as exe:
        print("Client disconnected",exe)
        sender.cancel()
        connections[room_id].discard(websocket)
        ws_queues[room_id].pop(websocket, None)
        left_user = user_ids[room_id].pop(websocket, None)

        if left_user:
            leave_message = {
                "type": "left_call",
                "user_id": left_user,
                "user_count": len(connections[room_id]),
                "users": list(user_ids[room_id].values()),
            }


            for ws in connections[room_id].copy():
                try:
                    await ws.send_json(leave_message)
                except:
                    connections[room_id].remove(ws)
                    user_ids[room_id].pop(ws, None)
                    
        if len(connections[room_id]) == 0:
            stop_flags[room_id].set()
            listeners[room_id].join()
            del listeners[room_id]
            del stop_flags[room_id]
            del connections[room_id]
            user_ids.pop(room_id, None)
            room_hosts.pop(room_id, None)
