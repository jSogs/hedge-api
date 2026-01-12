from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Union
import httpx
import os

router = APIRouter()

class SearchRequest(BaseModel):
    query: str
    limit: Union[str, int] = "auto"  # Can be "auto" or a number
    includeClosed: bool = False  # Only live markets for hedging
    minVolume: float = 0

@router.post("/search")
async def search(req: SearchRequest):
    """Search prediction markets using Probable API"""
    api_key = os.getenv("PROBABLE_API_KEY")
    api_url = os.getenv("PROBABLE_API_URL", "https://probable-api-app-d4a064dc7b26.herokuapp.com/api/search")
    
    if not api_key:
        raise HTTPException(status_code=500, detail="PROBABLE_API_KEY not configured")
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                api_url,
                headers={
                    "x-api-key": api_key,
                    "Content-Type": "application/json"
                },
                json={
                    "query": req.query,
                    "limit": req.limit,
                    "includeClosed": req.includeClosed,
                    "minVolume": req.minVolume
                },
                timeout=15.0
            )
            
            if response.status_code != 200:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Probable API error: {response.text}"
                )
            
            data = response.json()
            
            # Transform to match our expected format
            markets = data.get("markets", [])
            transformed_results = []
            
            for market in markets:
                result = {
                    "event_id": market.get("id"),
                    "event_title": market.get("title"),
                    "similarity": market.get("quality_score", 0) / 10.0,
                    "series_ticker": market.get("slug"),
                    "platform": market.get("platform"),
                    "category": market.get("category"),
                    "markets": [{
                        "market_id": market.get("id"),
                        "market_title": market.get("title"),
                        "external_market_id": market.get("external_id"),
                        "platform": market.get("platform"),
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
            
            return {
                "query": req.query,
                "results": transformed_results,
            }
            
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Search request timed out")
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Search service unavailable: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
