from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from app.db.supabase import sb

router = APIRouter(prefix="/v1/notifications", tags=["notifications"])

class Notification(BaseModel):
    id: str
    user_id: str
    type: str
    title: str
    message: Optional[str] = None
    recommendation_id: Optional[str] = None
    news_event_id: Optional[str] = None
    read_at: Optional[str] = None
    created_at: str

@router.get("/{user_id}", response_model=List[Notification])
def get_notifications(user_id: str, unread_only: bool = False, limit: int = 50):
    """Get notifications for a user"""
    query = sb.table("notifications").select("*").eq("user_id", user_id)
    
    if unread_only:
        query = query.is_("read_at", None)
    
    result = query.order("created_at", desc=True).limit(limit).execute()
    return result.data or []

@router.post("/{notification_id}/read")
def mark_as_read(notification_id: str):
    """Mark a notification as read"""
    result = sb.table("notifications").update({
        "read_at": datetime.utcnow().isoformat()
    }).eq("id", notification_id).execute()
    
    if not result.data:
        raise HTTPException(status_code=404, detail="Notification not found")
    
    return {"status": "read", "notification_id": notification_id}

@router.post("/{user_id}/read-all")
def mark_all_as_read(user_id: str):
    """Mark all notifications as read for a user"""
    result = sb.table("notifications").update({
        "read_at": datetime.utcnow().isoformat()
    }).eq("user_id", user_id).is_("read_at", None).execute()
    
    return {"status": "read", "count": len(result.data or [])}

@router.get("/{user_id}/unread-count")
def get_unread_count(user_id: str):
    """Get count of unread notifications for a user"""
    try:
        result = sb.table("notifications").select("*", count="exact").eq("user_id", user_id).is_("read_at", None).execute()
        return {"unread_count": result.count or 0}
    except Exception as e:
        print(f"Error getting unread count: {e}")
        # Return 0 if there's an error (table might not exist yet)
        return {"unread_count": 0}

