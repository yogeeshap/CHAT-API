import uuid
from typing import Optional,Any
from pydantic import BaseModel
from datetime import datetime

# Schemas
class User(BaseModel):
    user_id: str
    username: str
    email: str

class Room(BaseModel):
    room_id: str
    name: str
    created_by: str

class RoomMember(BaseModel):
    room_id: str
    user_id: str
    role: str
    
class Post(BaseModel):
    room_id: str
    messages: str
    author_id: str
    timestamp: Optional[datetime] = None
    edited_at: Optional[datetime] = None
    reactions: Optional[dict] = None

class Like(BaseModel):
    room_id: str
    post_id: str
    user_id: str

class RoomUser(BaseModel):
    room_name: str
    users: list

class RoomUserMap(BaseModel):
    room_name: Optional[str] = None
    username: Optional[str] = None
    email:Optional[str] = None
    user_id: Optional[str] = None
    room_id: Optional[str] = None
    room_ref: Optional[Any] = None
    password: Optional[str] = None
    phone_number:Optional[str] = None
    limit:Optional[int] = None

class UserModel(BaseModel):
    user_id: Optional[str] = None
    username: Optional[str] = None
    email: Optional[str] = None
    password: Optional[str] = None
    phone_number:Optional[str] = None

class VerifyOtpRequest(BaseModel):
    phone: str
    otp: str
    
class RoomUpdatePayload(BaseModel):
    name: str

class AddUserToRoom(BaseModel):
    users: list[str]
 