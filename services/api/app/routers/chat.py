from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import os

from openai import OpenAI
from app.db.supabase import sb
from app.ai.embed import embed_text

router = APIRouter(prefix="/v1/chat", tags=["chat"])
oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatMessage(BaseModel):
    conversation_id: Optional[str] = None
    message: str
    user_id: str

class ChatResponse(BaseModel):
    conversation_id: str
    message_id: str
    response: str
    markets: Optional[List[Dict[str, Any]]] = None
    
class ConversationSummary(BaseModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int

class ConversationDetail(BaseModel):
    id: str
    user_id: str
    title: str
    created_at: str
    updated_at: str
    messages: List[Dict[str, Any]]

# Tool functions that the LLM can call
def search_markets(query: str, limit: int = 5) -> List[Dict[str, Any]]:
    """Search for markets using semantic search"""
    try:
        q_emb = embed_text(query)
        
        payload = sb.rpc("search_kalshi_events_with_markets", {
            "query_embedding": q_emb,
            "match_count": limit,
            "markets_per_event": 3,
        }).execute().data
        
        payload = payload or {"results": []}
        results = payload.get("results") or []
        
        # Filter by minimum similarity
        filtered_results = [
            r for r in results
            if isinstance(r, dict) and r.get("similarity", 0) >= 0.5
        ]
        
        return filtered_results
    except Exception as e:
        print(f"Error in search_markets: {e}")
        return []

def get_user_profile(user_id: str) -> Optional[Dict[str, Any]]:
    """Get user's risk profile"""
    try:
        result = sb.table("profiles").select("*").eq("user_id", user_id).single().execute()
        return result.data
    except:
        return None

def get_market_details(market_id: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a specific market"""
    try:
        # Get market with event and outcomes
        market = sb.table("markets").select(
            "*, kalshi_events(*), market_outcomes(*)"
        ).eq("id", market_id).single().execute()
        return market.data
    except Exception as e:
        print(f"Error getting market details: {e}")
        return None

# Define tools for OpenAI function calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_markets",
            "description": "Search for prediction markets related to a topic. Use this when the user asks about specific markets, events, or hedging opportunities.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g., 'inflation', 'gas prices', 'interest rates')"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 5
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_market_details",
            "description": "Get detailed information about a specific market by ID. Use when user asks for more details about a specific market.",
            "parameters": {
                "type": "object",
                "properties": {
                    "market_id": {
                        "type": "string",
                        "description": "The UUID of the market"
                    }
                },
                "required": ["market_id"]
            }
        }
    }
]

def build_system_prompt(profile: Optional[Dict[str, Any]] = None) -> str:
    """Build system prompt with optional user profile context"""
    base_prompt = """You are Hedge, an AI assistant that helps users understand and navigate prediction markets for hedging purposes.

Your role:
- Help users understand hedging concepts and strategies
- Search for relevant prediction markets when asked
- Explain how specific markets work and their hedging potential
- Provide personalized recommendations based on user's risk profile
- Answer questions about markets, events, and pricing

Important guidelines:
- This is about hedging and risk management, NOT gambling or speculation
- Be clear, educational, and helpful
- When discussing specific markets, explain WHY they're relevant for hedging
- Use the search_markets function when users ask about specific topics or markets
- Keep responses concise but informative
"""
    
    if profile:
        profile_context = f"""

User Profile Context:
- Location: {profile.get('region', 'Unknown')}
- Industry: {profile.get('industry', 'Not specified')}
- Risk Horizon: {profile.get('risk_horizon', '90d')}
- Risk Style: {profile.get('risk_style', 'balanced')}
- Budget: ${profile.get('hedge_budget_monthly', 0)}/month
- Sensitivities: {', '.join(profile.get('sensitivities', []))}

Tailor your responses to their specific risk profile and concerns.
"""
        base_prompt += profile_context
    
    return base_prompt

def get_conversation_history(conversation_id: str, limit: int = 10) -> List[Dict[str, str]]:
    """Get recent messages from a conversation"""
    try:
        messages = sb.table("chat_messages").select(
            "role, content"
        ).eq("conversation_id", conversation_id).order(
            "created_at", desc=False
        ).limit(limit).execute()
        
        return [
            {"role": msg["role"], "content": msg["content"]}
            for msg in (messages.data or [])
        ]
    except Exception as e:
        print(f"Error getting conversation history: {e}")
        return []

def create_or_get_conversation(user_id: str, conversation_id: Optional[str] = None) -> str:
    """Create a new conversation or return existing one"""
    if conversation_id:
        # Verify conversation exists and belongs to user
        try:
            conv = sb.table("conversations").select("id").eq(
                "id", conversation_id
            ).eq("user_id", user_id).single().execute()
            
            if conv.data:
                return conversation_id
        except:
            pass
    
    # Create new conversation
    result = sb.table("conversations").insert({
        "user_id": user_id,
        "title": "New Chat",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat()
    }).execute()
    
    return result.data[0]["id"]

def update_conversation_title(conversation_id: str, first_message: str):
    """Auto-generate conversation title from first message"""
    # Simple title: first 50 chars of message
    title = first_message[:50] + ("..." if len(first_message) > 50 else "")
    
    sb.table("conversations").update({
        "title": title,
        "updated_at": datetime.utcnow().isoformat()
    }).eq("id", conversation_id).execute()

@router.post("/message", response_model=ChatResponse)
def send_message(payload: ChatMessage):
    """Send a message in a conversation and get AI response"""
    
    # Get or create conversation
    conversation_id = create_or_get_conversation(payload.user_id, payload.conversation_id)
    
    # Get user profile for context
    profile = get_user_profile(payload.user_id)
    
    # Get conversation history
    history = get_conversation_history(conversation_id, limit=10)
    
    # Build messages for OpenAI
    messages = [
        {"role": "system", "content": build_system_prompt(profile)}
    ]
    
    # Add conversation history
    messages.extend(history)
    
    # Add new user message
    messages.append({"role": "user", "content": payload.message})
    
    # Save user message to DB
    user_msg_result = sb.table("chat_messages").insert({
        "conversation_id": conversation_id,
        "role": "user",
        "content": payload.message,
        "created_at": datetime.utcnow().isoformat()
    }).execute()
    
    # Update conversation title if this is the first message
    if len(history) == 0:
        update_conversation_title(conversation_id, payload.message)
    
    # Call OpenAI with function calling
    try:
        response = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0.7,
        )
        
        assistant_message = response.choices[0].message
        markets_data = None
        
        # Handle function calls
        if assistant_message.tool_calls:
            # Process tool calls
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "search_markets":
                    markets_data = search_markets(**function_args)
                elif function_name == "get_market_details":
                    market_details = get_market_details(**function_args)
                    # Add market details to context for next response
                    
            # Get final response after function calls
            messages.append(assistant_message)
            
            # Add function results
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                if function_name == "search_markets":
                    result = search_markets(**function_args)
                elif function_name == "get_market_details":
                    result = get_market_details(**function_args)
                else:
                    result = None
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result) if result else "No results found"
                })
            
            # Get final response
            final_response = oai.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.7,
            )
            
            assistant_content = final_response.choices[0].message.content
        else:
            assistant_content = assistant_message.content
        
        # Save assistant response to DB
        response_data = {
            "markets": markets_data
        } if markets_data else None
        
        assistant_msg_result = sb.table("chat_messages").insert({
            "conversation_id": conversation_id,
            "role": "assistant",
            "content": assistant_content,
            "response_data": response_data,
            "created_at": datetime.utcnow().isoformat()
        }).execute()
        
        # Update conversation timestamp
        sb.table("conversations").update({
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", conversation_id).execute()
        
        return ChatResponse(
            conversation_id=conversation_id,
            message_id=assistant_msg_result.data[0]["id"],
            response=assistant_content,
            markets=markets_data
        )
        
    except Exception as e:
        print(f"Error in chat: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.get("/conversations/{user_id}", response_model=List[ConversationSummary])
def get_user_conversations(user_id: str, limit: int = 20):
    """Get list of user's conversations"""
    try:
        conversations = sb.table("conversations").select(
            "id, title, created_at, updated_at"
        ).eq("user_id", user_id).order(
            "updated_at", desc=True
        ).limit(limit).execute()
        
        result = []
        for conv in (conversations.data or []):
            # Count messages
            msg_count = sb.table("chat_messages").select(
                "id", count="exact"
            ).eq("conversation_id", conv["id"]).execute()
            
            result.append(ConversationSummary(
                id=conv["id"],
                title=conv["title"],
                created_at=conv["created_at"],
                updated_at=conv["updated_at"],
                message_count=msg_count.count or 0
            ))
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: str):
    """Get full conversation history"""
    try:
        # Get conversation
        conv = sb.table("conversations").select("*").eq(
            "id", conversation_id
        ).single().execute()
        
        if not conv.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get messages
        messages = sb.table("chat_messages").select("*").eq(
            "conversation_id", conversation_id
        ).order("created_at", desc=False).execute()
        
        return ConversationDetail(
            id=conv.data["id"],
            user_id=conv.data["user_id"],
            title=conv.data["title"],
            created_at=conv.data["created_at"],
            updated_at=conv.data["updated_at"],
            messages=messages.data or []
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str, user_id: str):
    """Delete a conversation and all its messages"""
    try:
        # Verify ownership
        conv = sb.table("conversations").select("id").eq(
            "id", conversation_id
        ).eq("user_id", user_id).single().execute()
        
        if not conv.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Delete messages first (FK constraint)
        sb.table("chat_messages").delete().eq(
            "conversation_id", conversation_id
        ).execute()
        
        # Delete conversation
        sb.table("conversations").delete().eq("id", conversation_id).execute()
        
        return {"status": "deleted", "conversation_id": conversation_id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.patch("/{conversation_id}/title")
def update_title(conversation_id: str, title: str, user_id: str):
    """Update conversation title"""
    try:
        result = sb.table("conversations").update({
            "title": title,
            "updated_at": datetime.utcnow().isoformat()
        }).eq("id", conversation_id).eq("user_id", user_id).execute()
        
        if not result.data:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        return {"status": "updated", "title": title}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
