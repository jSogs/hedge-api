"""Process newly added events: embed them and check for user recommendations"""
import os
import sys
import json
import time
from datetime import datetime, timedelta, UTC
from typing import List, Dict, Optional
from dotenv import load_dotenv
from supabase import create_client
from openai import OpenAI

# Add parent directory to path to import app modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

load_dotenv()

sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
oai = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

EMBED_MODEL = "text-embedding-3-small"

def build_event_text(e: Dict) -> str:
    """Build text for embedding from event data"""
    parts = [
        e.get("title") or "",
        e.get("subtitle") or "",
        e.get("description") or "",
        f"category: {e.get('category') or ''}",
        f"series: {e.get('series_ticker') or ''}",
        f"region: {e.get('region') or ''}",
    ]
    return "\n".join([p.strip() for p in parts if p and p.strip()])

def embed_new_events(event_ids: List[str] = None) -> List[str]:
    """Embed events that are missing embeddings"""
    from scripts.backfill_event_embeddings import (
        fetch_events_missing_embeddings,
        update_event_embedding
    )
    
    print("Embedding new events...")
    
    if event_ids:
        # Embed specific events
        events_to_embed = (
            sb.table("kalshi_events")
            .select("id,title,subtitle,description,category,series_ticker,region")
            .in_("id", event_ids)
            .is_("embedding", None)
            .execute()
            .data or []
        )
    else:
        # Embed all events missing embeddings
        events_to_embed = fetch_events_missing_embeddings(limit=100)
    
    if not events_to_embed:
        print("No events need embedding")
        return []
    
    texts = [build_event_text(e) for e in events_to_embed]
    
    # Batch embed
    try:
        emb = oai.embeddings.create(model=EMBED_MODEL, input=texts)
        vectors = [d.embedding for d in emb.data]
        
        for e, v in zip(events_to_embed, vectors):
            update_event_embedding(e["id"], v)
        
        print(f"✓ Embedded {len(events_to_embed)} events")
        return [e["id"] for e in events_to_embed]
    except Exception as e:
        print(f"Error embedding events: {e}")
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

def create_recommendation_for_new_event(
    user_id: str,
    profile: Dict,
    new_events: List[Dict],
    event_ids: List[str]
) -> Optional[str]:
    """Create recommendations for a user based on newly added events"""
    from app.routers.recommendations import build_profile_query, embed_text_1536
    
    if not new_events:
        return None
    
    # Build query from profile
    query_used = build_profile_query(profile)
    query_embedding = embed_text_1536(query_used)
    
    # Get market outcome map
    market_outcome_map = build_market_outcome_map(new_events)
    
    # Use LLM to select best hedges
    profile_type = (profile.get("profile_type") or "").lower()
    is_individual = profile_type == "individual"
    
    if is_individual:
        system = (
            "You are Hedge, a personal risk analyst for individuals. "
            "New prediction market events have been added that may be relevant for hedging. "
            "This is not gambling - help protect personal living costs. "
            "Only use the provided candidates; never invent markets or facts. "
            "Choose DISTINCT event_id values (no repeats) and EXACTLY ONE market per event. "
            "Choose hedge_leg YES/NO as the protective side and explain why in 1-2 sentences. "
            "Return STRICT JSON."
        )
    else:
        system = (
            "You are Hedge, a risk analyst for businesses. "
            "New prediction market events have been added that may be relevant for hedging. "
            "Only use the provided candidates; never invent markets or facts. "
            "IMPORTANT: choose DISTINCT event_id values (no repeats). "
            "For each selected event, choose EXACTLY ONE market under that event. "
            "Choose hedge_leg YES/NO and explain why in 1-2 sentences. "
            "Return STRICT JSON."
        )
    
    event_series = {e["event_id"]: e.get("series_ticker") for e in new_events}
    
    llm_input = {
        "profile": {
            "region": profile.get("region"),
            "industry": profile.get("industry"),
            "risk_horizon": profile.get("risk_horizon"),
            "risk_style": profile.get("risk_style"),
            "hedge_budget_monthly": float(profile.get("hedge_budget_monthly") or 0),
            "sensitivities": profile.get("sensitivities") or [],
            "profile_json": profile.get("profile_json") or {},
        },
        "candidates": new_events[:10],  # Limit candidates
        "limit": 3,  # Fewer recommendations per new event batch
        "output_schema": {
            "recommendations": [{
                "event_id": "uuid",
                "event_title": "string",
                "market_id": "uuid",
                "market_title": "string",
                "external_market_id": "string|null",
                "hedge_leg": "YES|NO",
                "why": "1-2 sentences explaining why this hedge is relevant",
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
                "status": r.get("status", "hedge_now"),
                "rationale": r.get("why", ""),
                "rec_json": {**r, "triggered_by_new_event": True},
                "series_ticker": event_series.get(r.get("event_id"))
            }
            rows.append(row_data)
        
        if rows:
            result = sb.table("recommendations").insert(rows).execute()
            rec_ids = [r.get("id") for r in (result.data or [])]
            
            # Create notification
            notification = {
                "user_id": user_id,
                "type": "new_recommendation",
                "title": f"New hedge opportunities available",
                "message": f"We've identified {len(recs)} new hedging opportunity(ies) that match your risk profile.",
                "recommendation_id": rec_ids[0] if rec_ids else None,
            }
            sb.table("notifications").insert(notification).execute()
            
            return rec_ids[0] if rec_ids else None
    
    except Exception as e:
        print(f"  Error creating recommendation: {e}")
        return None

def check_new_recommendations(event_ids: List[str] = None, hours: int = 1):
    """
    Check if users should get recommendations for newly added events.
    If event_ids is None, checks events added in the last N hours.
    """
    from app.routers.recommendations import build_profile_query, embed_text_1536
    
    # Get events to check
    if event_ids:
        events_to_check = (
            sb.table("kalshi_events")
            .select("id")
            .in_("id", event_ids)
            .execute()
            .data or []
        )
        event_ids_set = {e["id"] for e in events_to_check}
    else:
        # Check recently added events
        cutoff_time = (datetime.now(UTC) - timedelta(hours=hours)).isoformat()
        events_to_check = (
            sb.table("kalshi_events")
            .select("id")
            .gte("created_at", cutoff_time)
            .execute()
            .data or []
        )
        event_ids_set = {e["id"] for e in events_to_check}
    
    if not event_ids_set:
        print(f"No new events found (last {hours} hours)")
        return 0
    
    print(f"Checking {len(event_ids_set)} new events for user recommendations...")
    
    # Get all profiles
    profiles = sb.table("profiles").select("*").execute().data or []
    
    if not profiles:
        print("No user profiles found")
        return 0
    
    print(f"Checking against {len(profiles)} user profiles...")
    
    recommendations_created = 0
    
    for i, profile in enumerate(profiles):
        user_id = profile.get("user_id")
        if not user_id:
            continue
        
        # Add delay to avoid rate limits
        if i > 0:
            time.sleep(0.5)
        
        try:
            # Build profile query and embed
            query_used = build_profile_query(profile)
            query_embedding = embed_text_1536(query_used)
            
            # Search for matching events
            rpc = sb.rpc("search_kalshi_events_with_markets", {
                "query_embedding": query_embedding,
                "match_count": 20,
                "markets_per_event": 3,
            }).execute()
            
            rpc_json = rpc.data or {}
            candidates = rpc_json.get("results", []) if isinstance(rpc_json, dict) else []
            
            # Filter to only new events with good similarity
            new_candidates = [
                c for c in candidates 
                if c.get("event_id") in event_ids_set 
                and c.get("similarity", 0) >= 0.6
            ]
            
            if new_candidates:
                print(f"  User {user_id[:8]}...: Found {len(new_candidates)} new matching events")
                rec_id = create_recommendation_for_new_event(
                    user_id, profile, new_candidates, list(event_ids_set)
                )
                if rec_id:
                    recommendations_created += 1
                
        except Exception as e:
            print(f"  Error checking recommendations for user {user_id}: {e}")
            continue
    
    return recommendations_created

def main(event_ids: List[str] = None, hours: int = 1):
    """
    Main function to process new events.
    
    Args:
        event_ids: Specific event IDs to process (optional)
        hours: If event_ids not provided, check events from last N hours
    """
    print("=" * 60)
    print("Processing new events...")
    print("=" * 60)
    
    # 1. Embed new events
    embedded_ids = embed_new_events(event_ids)
    
    # 2. Check for recommendations
    # Use embedded_ids if we just embedded, otherwise use provided event_ids
    events_to_check = embedded_ids if embedded_ids else (event_ids if event_ids else None)
    
    if events_to_check:
        recommendations_created = check_new_recommendations(events_to_check, hours=hours)
        print(f"\n✓ Created {recommendations_created} new recommendation sets")
    else:
        print("\nNo events to check for recommendations")
    
    print("=" * 60)
    print("Done!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process new events and check for recommendations")
    parser.add_argument("--event-ids", nargs="+", help="Specific event IDs to process")
    parser.add_argument("--hours", type=int, default=1, help="Check events from last N hours (default: 1)")
    
    args = parser.parse_args()
    
    main(event_ids=args.event_ids, hours=args.hours)

