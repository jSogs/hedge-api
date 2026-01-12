from fastapi import APIRouter, HTTPException
import uuid
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
import json
import os
import httpx

from openai import OpenAI
from app.db.supabase import sb

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
def search_markets(query: str, limit: str = "auto") -> List[Dict[str, Any]]:
    """Search for markets using Probable API - natural language search.
    
    Args:
        query: Natural language search query
        limit: Number of results (default: "auto" - let LLM decide)
    """
    try:
        debug = os.getenv("PROBABLE_DEBUG", "").lower() in ("1", "true", "yes", "on")
        api_key = os.getenv("PROBABLE_API_KEY")
        api_url = os.getenv("PROBABLE_API_URL", "https://probable-api-app-d4a064dc7b26.herokuapp.com/api/search")
        
        if not api_key:
            print("Warning: PROBABLE_API_KEY not set, falling back to empty results")
            return []
        
        # Call Probable API
        if debug:
            print(f"[search_markets] Calling Probable API with query: {query}, limit: {limit}")
        response = httpx.post(
            api_url,
            headers={
                "x-api-key": api_key,
                "Content-Type": "application/json"
            },
            json={
                "query": query,
                "limit": limit,  # Can be "auto" or a number
                "includeClosed": False,  # Only live markets for hedging
                "minVolume": 0
            },
            timeout=30.0  # Increased timeout to 30 seconds
        )
        if debug:
            print(f"[search_markets] Got response: status={response.status_code}")
        
        if response.status_code != 200:
            print(f"Probable API error: {response.status_code} - {response.text}")
            return []
        
        data = response.json()
        
        # Transform Probable API response to our internal format
        markets = data.get("markets", [])
        
        # Convert to our expected format
        transformed_results = []
        for market in markets:
            # Group by market since Probable API returns individual markets
            result = {
                "event_id": market.get("id"),
                "event_title": market.get("title"),
                "similarity": market.get("quality_score", 0) / 10.0,  # Normalize quality_score to 0-1
                "series_ticker": market.get("slug"),
                "markets": [{
                    "market_id": market.get("id"),
                    "market_title": market.get("title"),
                    "external_market_id": market.get("external_id"),
                    "platform": market.get("platform"),
                    "category": market.get("category"),
                    "description": market.get("description"),
                    "end_time": market.get("end_time"),
                    "volume": market.get("volume_total"),
                    "liquidity": market.get("liquidity"),
                    "outcomes": [
                        {
                            "label": outcome,
                            "outcome_id": f"{market.get('id')}_{outcome.lower()}",
                            "latest_price": {
                                "price": float(market.get("outcomeprices", [])[idx]) if idx < len(market.get("outcomeprices", [])) and market.get("outcomeprices", [])[idx] else None,
                            } if idx < len(market.get("outcomeprices", [])) else None
                        }
                        for idx, outcome in enumerate(market.get("outcomes", []))
                    ]
                }]
            }
            transformed_results.append(result)
        
        return transformed_results
        
    except httpx.TimeoutException:
        print("Probable API timeout")
        return []
    except Exception as e:
        print(f"Error in search_markets: {e}")
        import traceback
        traceback.print_exc()
        return []

def _should_prefetch_markets(user_message: str) -> bool:
    """
    Heuristic: if user is asking to fetch/search markets or hedge opportunities, we prefetch
    markets up-front so the UI reliably shows market cards (without relying on the model to tool-call).
    """
    m = (user_message or "").lower()
    keywords = [
        "market", "markets",
        "search", "find",
        "fetch",
        "opportunit",  # opportunity/opportunities
        "hedge", "hedges", "hedging",
        "prediction market", "prediction markets",
        "kalshi", "polymarket",
    ]
    return any(k in m for k in keywords)

def _build_market_query(user_message: str) -> str:
    """
    Turn a user instruction like "fetch gas price hedges" into a cleaner natural-language
    query for the markets API.
    """
    raw = (user_message or "").strip()
    low = raw.lower()

    # A few high-signal shortcuts (works well for common cases)
    if "gas" in low and "price" in low:
        return "gas prices"
    if "steel" in low:
        return "steel prices"
    if "aluminum" in low or "aluminium" in low:
        return "aluminum prices"
    if "copper" in low:
        return "copper prices"
    if "lumber" in low:
        return "lumber prices"

    # Strip common command-y filler words
    stop_phrases = [
        "fetch", "search", "find", "show", "get", "give",
        "me", "some", "more",
        "markets", "market",
        "hedge", "hedges", "hedging",
        "opportunity", "opportunities",
        "please",
    ]
    cleaned = low
    for p in stop_phrases:
        cleaned = cleaned.replace(p, " ")
    cleaned = " ".join(cleaned.split()).strip()

    return cleaned or raw

def _tokenize_for_relevance(text: str) -> List[str]:
    text = (text or "").lower()
    for ch in "()[]{}.,:;!?/\\\"'`":
        text = text.replace(ch, " ")
    parts = [p for p in text.split() if p]
    stop = {
        "the","a","an","and","or","to","of","in","on","for","with","by","at","from","as",
        "will","be","is","are","was","were","it","this","that","these","those",
        "above","below","over","under",
        "price","prices",  # keep query meaningful; we can treat price words as low-signal
        "market","markets","prediction","hedge","hedges","hedging",
        "fetch","search","find","show","get","give","me","some","more","please",
    }
    return [p for p in parts if p not in stop and len(p) > 2]

def _filter_relevant_results(query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Hard filter: only keep results that mention at least one salient query token
    in the title/description. This prevents obviously unrelated markets from showing.
    """
    q_tokens = set(_tokenize_for_relevance(query))
    if not q_tokens:
        return results

    kept: List[Dict[str, Any]] = []
    for r in results or []:
        title = (r.get("event_title") or "") + " " + ((r.get("markets") or [{}])[0].get("market_title") or "")
        desc = ((r.get("markets") or [{}])[0].get("description") or "")
        hay = f"{title} {desc}".lower()
        if any(t in hay for t in q_tokens):
            kept.append(r)
    return kept

def _refine_query_once(query: str) -> str:
    """
    Basic refinement: prefer commodity-specific phrasing.
    """
    q = (query or "").strip().lower()
    if not q:
        return query
    if "steel" in q:
        return "steel prices commodity"
    if "copper" in q:
        return "copper prices commodity"
    if "aluminum" in q or "aluminium" in q:
        return "aluminum prices commodity"
    return query

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
            "description": "Search for prediction markets related to a topic. Use this when the user asks about specific markets, events, or hedging opportunities. Only returns live, open markets that users can trade.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query (e.g., 'inflation', 'gas prices', 'interest rates')"
                    },
                    "limit": {
                        "type": "string",
                        "description": "Number of results to return. Use 'auto' to let the system decide based on relevance, or specify a number if the user requests a specific amount.",
                        "default": "auto",
                        "enum": ["auto", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
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

def build_system_prompt(profile: Optional[Dict[str, Any]] = None, with_thinking: bool = False) -> str:
    """Build system prompt with optional user profile context"""
    
    thinking_instructions = """

IMPORTANT:
- Do NOT reveal your private chain-of-thought / internal reasoning. Don't write “I will…” planning text.
- Do NOT print tool call JSON or pretend-call tools in plain text.
- If you need market data, use the provided tools via tool-calling; otherwise respond normally.
""" if with_thinking else ""
    
    base_prompt = f"""You are Probable, an AI assistant that helps users understand and navigate prediction markets for hedging purposes.

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
- If markets have been fetched (tool results are present), you MUST base your answer on those markets:
  - Briefly say whether the fetched markets are suitable for the user's hedging goal.
  - If the fetched markets are not suitable (or the list is empty), ask a clarifying question AND suggest a better search angle (keywords/region/timeframe).
- Keep responses concise but informative{thinking_instructions}
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
    """Auto-generate contextual conversation title from first message using AI"""
    try:
        # Use GPT-4o-mini to generate a concise, contextual title
        response = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a title generator. Given a user's message, create a concise 2-5 word title that captures the topic/intent. Examples: 'hi' → 'Greeting', 'how can i hedge gas price risk?' → 'Gas Price Hedging', 'what markets are available for inflation?' → 'Inflation Markets'. Keep it short and descriptive."
                },
                {
                    "role": "user",
                    "content": first_message
                }
            ],
            temperature=0.3,
            max_tokens=15
        )
        
        title = response.choices[0].message.content.strip()
        
        # Fallback to first 50 chars if AI fails
        if not title or len(title) > 60:
            title = first_message[:50] + ("..." if len(first_message) > 50 else "")
    except Exception as e:
        print(f"Error generating title: {e}")
        # Fallback to first 50 chars
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

@router.post("/message/stream")
async def send_message_stream(payload: ChatMessage):
    """Send a message and stream the response with explicit thinking steps"""
    
    async def generate():
        try:
            # Get or create conversation
            conversation_id = create_or_get_conversation(payload.user_id, payload.conversation_id)
            
            # Get user profile
            profile = get_user_profile(payload.user_id)
            
            # Get conversation history
            history = get_conversation_history(conversation_id, limit=10)
            
            # Build messages with thinking-enabled prompt
            messages = [
                {"role": "system", "content": build_system_prompt(profile, with_thinking=True)}
            ]
            messages.extend(history)
            messages.append({"role": "user", "content": payload.message})
            
            # Save user message
            sb.table("chat_messages").insert({
                "conversation_id": conversation_id,
                "role": "user",
                "content": payload.message,
                "created_at": datetime.utcnow().isoformat()
            }).execute()
            
            # Update conversation title if first message
            if len(history) == 0:
                update_conversation_title(conversation_id, payload.message)
            
            # Send conversation_id first
            yield f"data: {json.dumps({'type': 'conversation_id', 'data': conversation_id})}\n\n"

            # Run the model in a loop so we can handle tool calls (e.g. search_markets) and continue streaming.
            # Without this, the stream "stops" when the model decides to call a tool.
            full_response = ""
            response_data: Optional[Dict[str, Any]] = None

            # Always provide at least one visible "thinking" step for UX, without leaking model internals.
            yield f"data: {json.dumps({'type': 'thinking_start'})}\n\n"
            yield f"data: {json.dumps({'type': 'thinking', 'content': 'Drafting response…'})}\n\n"
            yield f"data: {json.dumps({'type': 'thinking_end'})}\n\n"

            # Prefetch markets for "market search" prompts so we always have cards to show.
            # This also avoids cases where the model just *prints* a JSON blob instead of tool-calling.
            if _should_prefetch_markets(payload.message):
                prefetch_query = _build_market_query(payload.message)
                yield f"data: {json.dumps({'type': 'thinking_start'})}\n\n"
                yield f"data: {json.dumps({'type': 'thinking', 'content': f'Searching prediction markets for: {prefetch_query}'})}\n\n"
                yield f"data: {json.dumps({'type': 'thinking_end'})}\n\n"

                prefetched_raw = search_markets(query=prefetch_query, limit="auto")
                prefetched = _filter_relevant_results(prefetch_query, prefetched_raw)

                # If the search returns irrelevant results, refine once and retry.
                if len(prefetched) == 0 and len(prefetched_raw) > 0:
                    refined = _refine_query_once(prefetch_query)
                    if refined != prefetch_query:
                        yield f"data: {json.dumps({'type': 'thinking_start'})}\n\n"
                        yield f"data: {json.dumps({'type': 'thinking', 'content': f'Results didn’t match well; refining search to: {refined}'})}\n\n"
                        yield f"data: {json.dumps({'type': 'thinking_end'})}\n\n"
                        refined_raw = search_markets(query=refined, limit="auto")
                        prefetched = _filter_relevant_results(refined, refined_raw)
                        prefetch_query = refined

                response_data = {"query": prefetch_query, "results": prefetched}
                yield f"data: {json.dumps({'type': 'markets', 'query': prefetch_query, 'results': prefetched})}\n\n"

                # Provide the tool result to the model via tool messages so it can ground its answer.
                # We do NOT rely on the model to call tools for this initial search.
                # OpenAI requires tool_call_id length <= 40
                tool_call_id = f"pf_{uuid.uuid4().hex[:30]}"
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": "search_markets",
                            "arguments": json.dumps({"query": prefetch_query, "limit": "auto"})
                        }
                    }]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": json.dumps(prefetched),
                })

            def _normalize_tool_calls(tool_calls_delta: List[Any]) -> List[Dict[str, Any]]:
                """
                OpenAI tool_calls arrive chunked; we need to merge by index/id and concatenate arguments.
                Returns list of tool_calls in Chat Completions format.
                """
                merged: Dict[int, Dict[str, Any]] = {}
                for tc in tool_calls_delta:
                    idx = getattr(tc, "index", 0) or 0
                    if idx not in merged:
                        merged[idx] = {
                            "id": getattr(tc, "id", None),
                            "type": "function",
                            "function": {"name": None, "arguments": ""},
                        }
                    if getattr(tc, "id", None):
                        merged[idx]["id"] = getattr(tc, "id", None)
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        name = getattr(fn, "name", None)
                        args_part = getattr(fn, "arguments", None)
                        if name:
                            merged[idx]["function"]["name"] = name
                        if args_part:
                            merged[idx]["function"]["arguments"] += args_part

                # Drop any incomplete tool calls (no id or no name)
                normalized = []
                for _, call in sorted(merged.items(), key=lambda x: x[0]):
                    if call.get("id") and call["function"].get("name"):
                        normalized.append(call)
                return normalized

            while True:
                # Reset per-completion parsing state
                current_section = None  # 'thinking' or 'output'
                buffer = ""
                tool_calls_delta: List[Any] = []

                stream = oai.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=TOOLS,
                    tool_choice="auto",
                    temperature=0.7,
                    stream=True
                )

                for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if not delta:
                        continue

                    # Accumulate tool calls
                    if getattr(delta, "tool_calls", None):
                        tool_calls_delta.extend(delta.tool_calls)
                        continue

                    # Handle content
                    if getattr(delta, "content", None):
                        buffer += delta.content
                        full_response += delta.content

                        # Parse for thinking/output tags
                        while True:
                            if current_section is None:
                                thinking_start = buffer.find("<thinking>")
                                output_start = buffer.find("<output>")

                                if thinking_start != -1:
                                    if thinking_start > 0:
                                        text_before = buffer[:thinking_start]
                                        yield f"data: {json.dumps({'type': 'output', 'content': text_before})}\n\n"

                                    buffer = buffer[thinking_start + 10:]  # Remove <thinking>
                                    current_section = 'thinking'
                                    yield f"data: {json.dumps({'type': 'thinking_start'})}\n\n"
                                elif output_start != -1:
                                    if output_start > 0:
                                        text_before = buffer[:output_start]
                                        yield f"data: {json.dumps({'type': 'output', 'content': text_before})}\n\n"

                                    buffer = buffer[output_start + 8:]  # Remove <output>
                                    current_section = 'output'
                                else:
                                    break

                            elif current_section == 'thinking':
                                end_tag = buffer.find("</thinking>")
                                if end_tag != -1:
                                    thinking_content = buffer[:end_tag]
                                    if thinking_content:
                                        yield f"data: {json.dumps({'type': 'thinking', 'content': thinking_content})}\n\n"

                                    yield f"data: {json.dumps({'type': 'thinking_end'})}\n\n"
                                    buffer = buffer[end_tag + 11:]  # Remove </thinking>
                                    current_section = None
                                else:
                                    tail_len = 10  # "</thinking>" len 11, keep 10
                                    if len(buffer) > tail_len + 10:
                                        flush_upto = len(buffer) - tail_len
                                        yield f"data: {json.dumps({'type': 'thinking', 'content': buffer[:flush_upto]})}\n\n"
                                        buffer = buffer[flush_upto:]
                                    break

                            elif current_section == 'output':
                                end_tag = buffer.find("</output>")
                                if end_tag != -1:
                                    output_content = buffer[:end_tag]
                                    if output_content:
                                        yield f"data: {json.dumps({'type': 'output', 'content': output_content})}\n\n"

                                    buffer = buffer[end_tag + 9:]  # Remove </output>
                                    current_section = None
                                else:
                                    tail_len = 8  # "</output>" len 9, keep 8
                                    if len(buffer) > tail_len + 5:
                                        flush_upto = len(buffer) - tail_len
                                        yield f"data: {json.dumps({'type': 'output', 'content': buffer[:flush_upto]})}\n\n"
                                        buffer = buffer[flush_upto:]
                                    break

                # Send any remaining buffer at end of this completion
                if buffer:
                    content_type = 'thinking' if current_section == 'thinking' else 'output'
                    yield f"data: {json.dumps({'type': content_type, 'content': buffer})}\n\n"

                tool_calls = _normalize_tool_calls(tool_calls_delta)
                if not tool_calls:
                    break

                # Execute tool calls, append results, then loop to let the model continue.
                messages.append({"role": "assistant", "tool_calls": tool_calls})
                for call in tool_calls:
                    tool_name = call["function"]["name"]
                    tool_args_raw = call["function"].get("arguments", "") or "{}"
                    tool_call_id = call["id"]

                    try:
                        tool_args = json.loads(tool_args_raw) if tool_args_raw else {}
                    except Exception:
                        tool_args = {}

                    tool_result: Any = None
                    if tool_name == "search_markets":
                        q = tool_args.get("query") or ""
                        lim = tool_args.get("limit") or "auto"
                        tool_raw = search_markets(query=q, limit=lim)
                        tool_result = _filter_relevant_results(q, tool_raw)

                        # Surface results to the UI for market cards
                        response_data = {"query": q, "results": tool_result}
                        yield f"data: {json.dumps({'type': 'markets', 'query': q, 'results': tool_result})}\n\n"

                    elif tool_name == "get_market_details":
                        mid = tool_args.get("market_id") or ""
                        tool_result = get_market_details(market_id=mid)

                    else:
                        tool_result = {"error": f"Unknown tool: {tool_name}"}

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call_id,
                        "content": json.dumps(tool_result),
                    })

            # Clean up the full response (remove tags for storage)
            clean_response = full_response.replace("<thinking>", "").replace("</thinking>", "")
            clean_response = clean_response.replace("<output>", "").replace("</output>", "").strip()
            
            # Save assistant response to DB
            assistant_msg_result = sb.table("chat_messages").insert({
                "conversation_id": conversation_id,
                "role": "assistant",
                "content": clean_response,
                "response_data": response_data,
                "created_at": datetime.utcnow().isoformat()
            }).execute()
            
            # Update conversation timestamp
            sb.table("conversations").update({
                "updated_at": datetime.utcnow().isoformat()
            }).eq("id", conversation_id).execute()
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'done', 'message_id': assistant_msg_result.data[0]['id']})}\n\n"
            
        except Exception as e:
            print(f"Error in streaming chat: {e}")
            import traceback
            traceback.print_exc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

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
