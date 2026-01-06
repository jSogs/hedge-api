import os
import json
import sys
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Optional
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI
import requests

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

load_dotenv()

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBED_MODEL = "text-embedding-3-small"
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
NEWS_API_URL = "https://newsapi.org/v2/everything"

def fetch_recent_news(hours: int = 1) -> List[Dict]:
    """Fetch news from NewsAPI for the last N hours"""
    if not NEWS_API_KEY:
        print("Warning: NEWS_API_KEY not set. Skipping news fetch.")
        return []
    
    # Format date for NewsAPI (YYYY-MM-DDTHH:MM:SSZ, no microseconds)
    # from_date = (datetime.now(UTC) - timedelta(hours=hours)).replace(microsecond=0).isoformat().replace('+00:00', 'Z')
    from_date = (datetime.now(UTC) - timedelta(hours=hours)).strftime("%Y-%m-%d")
    # params = {
    #     "apiKey": NEWS_API_KEY,
    #     "country": "us",
    #     "category": "business",  # Options: business, general, technology, health, science, sports, entertainment
    # }
    params = {
        "apiKey": NEWS_API_KEY,
        "q": "world OR global OR economy OR politics OR finance OR business OR markets OR housing OR technology OR inflation OR interest rates OR unemployment",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 100,
        "from": from_date,
    }
    
    try:
        print(f"Fetching news from {from_date}...")
        print(f"Query: {params['q']}")
        response = requests.get(NEWS_API_URL, params=params, timeout=30)
        
        # Check status before raising
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error response: {response.text[:500]}")
            response.raise_for_status()
        
        data = response.json()
        articles = data.get("articles", [])
        total_results = data.get("totalResults", 0)
        status = data.get("status", "unknown")
        
        print(f"NewsAPI status: {status}")
        print(f"Total results available: {total_results}")
        print(f"Articles returned: {len(articles)}")
        
        if status == "error":
            print(f"NewsAPI error: {data.get('message', 'Unknown error')}")
            return []
        
        return articles
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error fetching news: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Response status: {e.response.status_code}")
            print(f"Response body: {e.response.text[:500]}")
        return []
    except Exception as e:
        print(f"Error fetching news: {e}")
        import traceback
        traceback.print_exc()
        return []

def classify_importance(news_item: Dict) -> Optional[float]:
    """Use LLM to classify if this is a significant world event (0-1 score)"""
    title = news_item.get("title", "")
    description = news_item.get("description", "")
    content = f"{title}\n\n{description}"[:1000]
    
    prompt = f"""Rate the importance of this news event for financial/economic hedging purposes on a scale of 0.0 to 1.0.

Consider:
- Impact on global markets, economy, or major industries
- Potential to affect prediction markets
- Relevance to risk hedging scenarios
- Scale and significance of the event

News:
{content}

Respond with ONLY a JSON object: {{"importance": 0.85, "reason": "brief explanation"}}
"""
    
    try:
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": "You are a financial news analyst. Return only valid JSON."},
                {"role": "user", "content": prompt}
            ]
        )
        result = json.loads(resp.choices[0].message.content)
        return result.get("importance", 0.0)
    except Exception as e:
        print(f"Error classifying news: {e}")
        return None

def embed_news(news_item: Dict) -> Optional[List[float]]:
    """Create embedding for news article"""
    text = f"{news_item.get('title', '')}\n{news_item.get('description', '')}"
    if not text.strip():
        return None
    
    try:
        emb = oai.embeddings.create(model=EMBED_MODEL, input=text.strip())
        return emb.data[0].embedding
    except Exception as e:
        print(f"Error embedding news: {e}")
        return None

def find_affected_kalshi_events(news_embedding: List[float], limit: int = 20) -> List[Dict]:
    """Find Kalshi events that might be affected by this news"""
    try:
        rpc = sb.rpc("search_kalshi_events_with_markets", {
            "query_embedding": news_embedding,
            "match_count": limit,
            "markets_per_event": 3,
        }).execute()
        
        rpc_json = rpc.data or {}
        results = rpc_json.get("results", []) if isinstance(rpc_json, dict) else []
        
        # Filter by minimum similarity
        filtered = [
            r for r in results
            if isinstance(r, dict) and r.get("similarity", 0) >= 0.6
        ]
        return filtered
    except Exception as e:
        print(f"Error finding affected events: {e}")
        return []

def build_market_outcome_map(candidates: List[Dict]) -> Dict:
    """Build map of market_id -> {YES: {outcome_id, price}, NO: {outcome_id, price}}"""
    market_outcome_map = {}
    
    for ev in candidates:
        for m in (ev.get("markets") or []):
            mid = m.get("market_id")
            if not mid:
                continue
            
            by_label = {}
            for o in (m.get("outcomes") or []):
                label = (o.get("label") or "").upper()
                oid = o.get("outcome_id")
                lp = o.get("latest_price") or {}
                price = lp.get("price")
                if label in ("YES", "NO") and oid:
                    by_label[label] = {"outcome_id": oid, "price": price}
            
            market_outcome_map[mid] = by_label
    
    return market_outcome_map

def create_recommendation_for_user(
    user_id: str,
    profile: Dict,
    news_event_id: str,
    affected_events: List[Dict],
    news_title: str
) -> Optional[str]:
    """Create a recommendation for a user based on news event"""
    from app.routers.recommendations import build_profile_query, embed_text_1536
    
    if not affected_events:
        return None
    
    # Build query combining profile + news context
    profile_query = build_profile_query(profile)
    news_context = f"Recent news: {news_title}"
    combined_query = f"{profile_query} | {news_context}"
    
    query_embedding = embed_text_1536(combined_query)
    
    # Get market outcome map
    market_outcome_map = build_market_outcome_map(affected_events)
    
    # Use LLM to select best hedge given news context
    profile_type = (profile.get("profile_type") or "").lower()
    is_individual = profile_type == "individual"
    
    if is_individual:
        system = (
            "You are Hedge, a personal risk analyst for individuals. "
            "A significant news event just occurred that may affect prediction markets. "
            "Recommend the best hedge for this individual given the news context. "
            "This is not gambling - help protect personal living costs. "
            "Only use the provided candidates; never invent markets or facts. "
            "Choose DISTINCT event_id values (no repeats) and EXACTLY ONE market per event. "
            "Choose hedge_leg YES/NO as the protective side and explain why in 1-2 sentences. "
            "Return STRICT JSON."
        )
    else:
        system = (
            "You are Hedge, a risk analyst for businesses. "
            "A significant news event just occurred that may affect prediction markets. "
            "Recommend the best hedge for this business given the news context. "
            "Only use the provided candidates; never invent markets or facts. "
            "IMPORTANT: choose DISTINCT event_id values (no repeats). "
            "For each selected event, choose EXACTLY ONE market under that event. "
            "Choose hedge_leg YES/NO and explain why in 1-2 sentences. "
            "Return STRICT JSON."
        )
    
    event_series = {e["event_id"]: e.get("series_ticker") for e in affected_events}
    
    llm_input = {
        "news_context": news_title,
        "profile": {
            "region": profile.get("region"),
            "industry": profile.get("industry"),
            "risk_horizon": profile.get("risk_horizon"),
            "risk_style": profile.get("risk_style"),
            "hedge_budget_monthly": float(profile.get("hedge_budget_monthly") or 0),
            "sensitivities": profile.get("sensitivities") or [],
            "profile_json": profile.get("profile_json") or {},
        },
        "candidates": affected_events[:10],  # Limit candidates
        "limit": 3,  # Fewer recommendations per news event
        "output_schema": {
            "recommendations": [{
                "event_id": "uuid",
                "event_title": "string",
                "market_id": "uuid",
                "market_title": "string",
                "external_market_id": "string|null",
                "hedge_leg": "YES|NO",
                "why": "1-2 sentences explaining why this hedge is relevant given the news",
                "status": "hedge_now|wait",
                "series_ticker": "series ticker from db"
            }]
        }
    }
    
    try:
        resp = oai.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(llm_input)},
            ],
        )
        
        parsed = json.loads(resp.choices[0].message.content)
        recs = parsed.get("recommendations", [])[:3]
        
        if not recs:
            return None
        
        # Deduplicate
        seen_events, seen_markets = set(), set()
        unique = []
        for r in recs:
            e, m = r.get("event_id"), r.get("market_id")
            if not e or not m:
                continue
            if e in seen_events or m in seen_markets:
                continue
            seen_events.add(e)
            seen_markets.add(m)
            unique.append(r)
        
        recs = unique[:3]
        
        # Persist recommendations
        profile_id = profile["id"]
        rows = []
        rec_ids = []
        
        for r in recs:
            market_id = r.get("market_id")
            hedge_leg = (r.get("hedge_leg") or "YES").upper()
            
            outcome_id = None
            price_now = None
            m = market_outcome_map.get(market_id, {})
            if hedge_leg in m:
                outcome_id = m[hedge_leg]["outcome_id"]
                price_now = m[hedge_leg]["price"]
            
            row_data = {
                "user_id": user_id,
                "profile_id": profile_id,
                "event_id": r.get("event_id"),
                "market_id": market_id,
                "outcome_id": outcome_id,
                "match_score": 0,
                "price_now": price_now,
                "price_threshold": None,
                "status": r.get("status", "hedge_now"),  # News-driven recommendations are urgent
                "rationale": r.get("why", ""),
                "rec_json": {**r, "triggered_by_news": True, "news_event_id": news_event_id},
                "series_ticker": event_series.get(r.get("event_id"))
            }
            rows.append(row_data)
        
        if rows:
            result = sb.table("recommendations").insert(rows).execute()
            rec_ids = [r.get("id") for r in (result.data or [])]
            
            # Link to news event
            if rec_ids and news_event_id:
                news_rec_links = [
                    {"news_event_id": news_event_id, "recommendation_id": rid}
                    for rid in rec_ids
                ]
                sb.table("news_event_recommendations").insert(news_rec_links).execute()
            
            # Create notification
            notification = {
                "user_id": user_id,
                "type": "new_recommendation",
                "title": f"New hedge recommendation: {news_title[:50]}",
                "message": f"A recent news event may affect your risk profile. We've identified {len(recs)} hedging opportunity(ies).",
                "recommendation_id": rec_ids[0] if rec_ids else None,
                "news_event_id": news_event_id,
            }
            sb.table("notifications").insert(notification).execute()
            
            return rec_ids[0] if rec_ids else None
    
    except Exception as e:
        print(f"Error creating recommendation: {e}")
        return None

def process_news_item(news_item: Dict) -> Optional[str]:
    """Process a single news item: classify, embed, match, and create recommendations"""
    
    # Check if we've already processed this news (by URL)
    url = news_item.get("url", "")
    title = news_item.get("title", "")
    
    if not title:
        print("  Skipping: No title")
        return None
    
    print(f"Processing: {title[:80]}")
    
    # Check for duplicates
    existing = sb.table("news_events").select("id").eq("url", url).execute().data
    if existing:
        print(f"  Already processed (duplicate)")
        return None
    
    # Classify importance
    print(f"  Classifying importance...")
    importance = classify_importance(news_item)
    if not importance:
        print(f"  Failed to classify importance")
        return None
    
    print(f"  Importance score: {importance:.2f} (threshold: 0.7)")
    
    if importance < 0.7:  # Threshold for "big" events
        print(f"  Not important enough - skipping")
        return None
    
    print(f"  ✓ Important enough - continuing processing")
    
    # Create embedding
    embedding = embed_news(news_item)
    if not embedding:
        return None
    
    # Store news event
    published_at = news_item.get("publishedAt")
    try:
        # Handle different date formats
        if published_at.endswith("Z"):
            published_dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        else:
            published_dt = datetime.fromisoformat(published_at)
    except:
        published_dt = datetime.now(UTC)
    
    news_row = {
        "title": title,
        "description": news_item.get("description", ""),
        "source": news_item.get("source", {}).get("name", "") if isinstance(news_item.get("source"), dict) else str(news_item.get("source", "")),
        "url": url,
        "published_at": published_dt.isoformat(),
        "embedding": embedding,
        "importance_score": importance,
        "news_json": news_item,
        "processed_at": datetime.now(UTC).isoformat(),
    }
    
    result = sb.table("news_events").insert(news_row).execute()
    news_event_id = result.data[0]["id"] if result.data else None
    
    if not news_event_id:
        return None
    
    # Find affected Kalshi events
    print(f"  Finding affected Kalshi events...")
    affected_events = find_affected_kalshi_events(embedding)
    if not affected_events:
        print(f"  No affected Kalshi events found (similarity < 0.6)")
        return news_event_id
    
    print(f"  Found {len(affected_events)} affected Kalshi events")
    
    # Find all active profiles
    profiles = sb.table("profiles").select("*").execute().data or []
    
    # For each user, check if they're affected and create recommendations
    created_count = 0
    for profile in profiles:
        user_id = profile.get("user_id")
        if not user_id:
            continue
        
        try:
            rec_id = create_recommendation_for_user(
                user_id, profile, news_event_id, affected_events, title
            )
            if rec_id:
                created_count += 1
        except Exception as e:
            print(f"Error creating recommendation for user {user_id}: {e}")
            continue
    
    print(f"Created {created_count} recommendations for {len(profiles)} users")
    return news_event_id

def main():
    """Main function to fetch and process news"""
    print("Fetching recent news...")
    news_items = fetch_recent_news(hours=24)
    
    if not news_items:
        print("No news items found")
        return
    
    print(f"Found {len(news_items)} news items")
    print("-" * 60)
    
    processed = 0
    skipped_low_importance = 0
    skipped_duplicates = 0
    skipped_no_matches = 0
    
    for item in news_items:
        try:
            news_id = process_news_item(item)
            if news_id:
                processed += 1
            else:
                # Count different skip reasons (this is approximate)
                skipped_low_importance += 1
        except Exception as e:
            print(f"Error processing news item: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("-" * 60)
    print(f"Summary:")
    print(f"  Processed: {processed} important news events")
    print(f"  Skipped: {skipped_low_importance} items (low importance, duplicates, or no matches)")

if __name__ == "__main__":
    main()

