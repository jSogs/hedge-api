# services/api/app/routers/recommendations.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import os, json

from openai import OpenAI
from app.db.supabase import sb  # <- uses your existing file

router = APIRouter(prefix="/v1/recommendations", tags=["recommendations"])
oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class RunRecommendationsReq(BaseModel):
    user_id: str  # uuid string
    limit: int = 10
    match_count: int = 20
    markets_per_event: int = 3

class HedgeRec(BaseModel):
    event_id: str
    event_title: str
    market_id: str
    market_title: str
    external_market_id: Optional[str] = None
    hedge_leg: str  # "YES" or "NO"
    why: str
    estimated_loss: Optional[float] = None  # Estimated loss if event happens (USD)
    estimated_recovery: Optional[float] = None  # Estimated recovery if hedge is purchased (USD)

class RunRecommendationsRes(BaseModel):
    user_id: str
    profile_id: str
    query_used: str
    recommendations: List[HedgeRec]
    candidates_used: int

def build_profile_query(profile: Dict[str, Any]) -> str:
    parts: List[str] = []
    parts.append(f"Region: {profile.get('region')}")
    if profile.get("industry"):
        parts.append(f"Industry: {profile.get('industry')}")
    if profile.get("risk_horizon"):
        parts.append(f"Horizon: {profile.get('risk_horizon')}")
    sens = profile.get("sensitivities") or []
    if sens:
        parts.append("Sensitivities: " + ", ".join(sens))
    pj = profile.get("profile_json") or {}
    if isinstance(pj, dict) and pj:
        parts.append("Notes: " + json.dumps(pj)[:500])
    return " | ".join(parts)

def embed_text_1536(text: str) -> List[float]:
    emb = oai.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return emb.data[0].embedding

@router.post("/run", response_model=RunRecommendationsRes)
def run_recommendations(payload: RunRecommendationsReq):
    # 1) Load profile by user_id (unique constraint)
    prof = (
        sb.table("profiles")
        .select("*")
        .eq("user_id", payload.user_id)
        .single()
        .execute()
    )

    profile = prof.data
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found for user_id")

    profile_id = profile["id"]

    # 2) Build query from profile and embed it
    query_used = build_profile_query(profile)
    query_embedding = embed_text_1536(query_used)

    # 3) Call your RPC (expects vector(1536))
    rpc = sb.rpc("search_kalshi_events_with_markets", {
        "query_embedding": query_embedding,
        "match_count": payload.match_count,
        "markets_per_event": payload.markets_per_event,
    }).execute()

    rpc_json = rpc.data or {}
    candidates = rpc_json.get("results", []) if isinstance(rpc_json, dict) else []
    
    market_outcome_map = {}

    for ev in candidates:
        for m in (ev.get("markets") or []):
            mid = m.get("market_id")
            if not mid:
                continue

            by_label = {}
            for o in (m.get("outcomes") or []):
                label = (o.get("label") or "").upper()   # YES/NO from market_outcomes.label
                oid = o.get("outcome_id")                # market_outcomes.id
                lp = o.get("latest_price") or {}
                price = lp.get("price")                  # from v_latest_outcome_price / market_prices
                if label in ("YES", "NO") and oid:
                    by_label[label] = {"outcome_id": oid, "price": price}

            market_outcome_map[mid] = by_label


    if not candidates:
        return {
            "user_id": payload.user_id,
            "profile_id": profile_id,
            "query_used": query_used,
            "recommendations": [],
            "candidates_used": 0,
        }

    # 4) Ask LLM to pick top hedges + explain
    profile_type = (profile.get("profile_type") or "").lower()
    is_individual = profile_type == "individual"

    if is_individual:
        system = (
            "You are Hedge, a personal risk analyst for individuals. "
            "This is not gambling. Your goal is to help protect personal living costs "
            "(rent, gas, groceries, utilities, inflation, interest rates). "
            "Use clear, plain language and focus on everyday impact. "
            "Only use the provided candidates; never invent markets or facts. "
            "Choose DISTINCT event_id values (no repeats) and EXACTLY ONE market per event. "
            "Choose hedge_leg YES/NO as the protective side for the individual and explain why in 1-2 sentences. "
            "For each recommendation, estimate: "
            "1) estimated_loss: The potential financial loss (in USD) the individual could face if this event happens, "
            "based on their profile (region, industry, sensitivities, hedge_budget_monthly). "
            "2) estimated_recovery: If they purchase contracts worth their hedge_budget_monthly (or a reasonable portion) "
            "at the current market price (found in the candidate's outcomes.latest_price), calculate how much they'd recover "
            "if the hedge_leg resolves. For binary markets: if price is P (0-1), spending $X buys $X/P worth of contracts, "
            "so if it resolves YES they'd recover $X/P. Return STRICT JSON."
        )
    else:
        system = (
            "You are Hedge, a risk analyst for businesses. "
            "Recommend prediction-market hedges (not gambling). "
            "Only use the provided candidates; never invent markets or facts. "
            "IMPORTANT: choose DISTINCT event_id values (no repeats). "
            "For each selected event, choose EXACTLY ONE market under that event. "
            "Choose hedge_leg YES/NO and explain why in 1-2 sentences. "
            "For each recommendation, estimate: "
            "1) estimated_loss: The potential financial loss (in USD) the business could face if this event happens, "
            "based on their profile (region, industry, sensitivities, hedge_budget_monthly). "
            "2) estimated_recovery: If they purchase contracts worth their hedge_budget_monthly (or a reasonable portion) "
            "at the current market price (found in the candidate's outcomes.latest_price), calculate how much they'd recover "
            "if the hedge_leg resolves. For binary markets: if price is P (0-1), spending $X buys $X/P worth of contracts, "
            "so if it resolves YES they'd recover $X/P. Return STRICT JSON."
        )

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
        "candidates": candidates,
        "limit": payload.limit,
        "output_schema": {
            "recommendations": [
                {
                    "event_id": "uuid",
                    "event_title": "string",
                    "market_id": "uuid",
                    "market_title": "string",
                    "external_market_id": "string|null",
                    "hedge_leg": "YES|NO",
                    "why": "1-2 sentences",
                    "status": "hedge_now|wait",
                    "estimated_loss": "number|null - Estimated loss in USD if event happens",
                    "estimated_recovery": "number|null - Estimated recovery in USD if hedge contracts are purchased at current price",
                }
            ]
        }
    }

    resp = oai.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0.2,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(llm_input)},
        ],
    )

    parsed = json.loads(resp.choices[0].message.content)
    recs = (parsed.get("recommendations") or [])[: payload.limit]

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

    recs = unique[: payload.limit]

    event_series = {e["event_id"]: e.get("series_ticker") for e in candidates}

    # Persist recommendations (one row per rec)
    rows = []
    for r in recs:
        market_id = r.get("market_id")
        hedge_leg = (r.get("hedge_leg") or "YES").upper()

        outcome_id = None
        price_now = None

        m = market_outcome_map.get(market_id, {})
        if hedge_leg in m:
            outcome_id = m[hedge_leg]["outcome_id"]
            price_now = m[hedge_leg]["price"]

        rows.append({
            "user_id": payload.user_id,
            "profile_id": profile_id,
            "event_id": r.get("event_id"),
            "market_id": market_id,
            "outcome_id": outcome_id,
            "match_score": 0,
            "price_now": price_now,
            "price_threshold": None,
            "status": r.get("status"),
            "rationale": r.get("why", ""),
            "rec_json": r,
            "estimated_loss": r.get("estimated_loss"),
            "estimated_recovery": r.get("estimated_recovery"),
        })

    if rows:
        sb.table("recommendations").insert(rows).execute()


    return {
        "user_id": payload.user_id,
        "profile_id": profile_id,
        "query_used": query_used,
        "recommendations": recs,
        "candidates_used": len(candidates),
    }
