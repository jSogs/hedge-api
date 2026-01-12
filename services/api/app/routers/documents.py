from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, List
from datetime import datetime, UTC
import json
import base64
import os
import io
import httpx

from openai import OpenAI
from app.db.supabase import sb

# Try to import PDF handling libraries
try:
    from pdf2image import convert_from_bytes
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    print("Warning: pdf2image not installed. PDF support disabled. Install with: pip install pdf2image")

router = APIRouter(prefix="/v1/profile", tags=["documents"])
oai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class DocumentAnalysis(BaseModel):
    status: str
    analysis: Dict[str, Any]

@router.post("/{user_id}/analyze-document", response_model=DocumentAnalysis)
async def analyze_financial_document(user_id: str, file: UploadFile = File(...)):
    """
    Upload and analyze a financial document (bank statement, earnings report, etc.)
    Uses GPT-4 Vision to extract insights and store in user profile
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith(('image/', 'application/pdf')):
        raise HTTPException(
            status_code=400, 
            detail="File must be an image (PNG, JPG) or PDF"
        )
    
    try:
        # Read file bytes
        file_bytes = await file.read()
        
        print(f"Analyzing document for user {user_id}...")
        print(f"File type: {file.content_type}, Size: {len(file_bytes)} bytes")
        
        # Handle PDFs - convert to images
        if file.content_type == 'application/pdf':
            if not PDF_SUPPORT:
                raise HTTPException(
                    status_code=400,
                    detail="PDF support not available. Please upload an image (PNG/JPG) instead."
                )
            
            print("Converting PDF to images...")
            # Convert PDF to images (use first page only for MVP)
            images = convert_from_bytes(file_bytes, first_page=1, last_page=1, dpi=200)
            
            if not images:
                raise HTTPException(status_code=400, detail="Could not extract images from PDF")
            
            # Convert PIL Image to bytes
            img_byte_arr = io.BytesIO()
            images[0].save(img_byte_arr, format='PNG')
            file_bytes = img_byte_arr.getvalue()
            mime_type = "image/png"
            print(f"✓ Converted PDF to PNG ({len(file_bytes)} bytes)")
        else:
            # It's already an image
            mime_type = file.content_type
        
        # Convert to base64 for GPT-4 Vision
        base64_image = base64.b64encode(file_bytes).decode('utf-8')
        
        print(f"Sending to GPT-4 Vision (mime: {mime_type})...")
        
        # Step 1: Extract financial data with GPT-4 Vision
        extraction_response = oai.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """Analyze this financial document and extract key information.

Extract:
1. Monthly or annual income (if visible)
2. Major expense categories with amounts (rent, mortgage, gas, groceries, utilities, insurance, etc.)
3. Time period covered
4. Any notable patterns or trends in spending

Return as JSON with this structure:
{
  "income": {
    "amount": number or null,
    "frequency": "monthly" or "annual" or null
  },
  "expenses": {
    "category_name": amount,
    ...
  },
  "period": {
    "start": "YYYY-MM" or null,
    "end": "YYYY-MM" or null
  },
  "total_expenses": number or null,
  "patterns": ["brief observation 1", "brief observation 2"]
}"""
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime_type};base64,{base64_image}"
                        }
                    }
                ]
            }],
            max_tokens=1500,
            response_format={"type": "json_object"}
        )
        
        extracted_data = json.loads(extraction_response.choices[0].message.content)
        print(f"Extracted data: {json.dumps(extracted_data, indent=2)}")
        
        # Step 2: Analyze vulnerabilities and generate prediction market queries
        analysis_response = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Based on this user's financial data:
{json.dumps(extracted_data, indent=2)}

Analyze their spending patterns and identify what they should hedge using PREDICTION MARKETS.

Look at their expense categories (gas, food, travel, retail, utilities, etc.) and identify:
1. Which expense categories are most significant or volatile?
2. What economic events or price changes would impact these expenses?
3. Generate 3-5 SHORT search queries (2-3 words each) to find relevant prediction markets.

Examples of GOOD market search queries:
- "gas prices" (if they spend on gas/fuel)
- "food inflation" (if they spend heavily on groceries)
- "airline prices" (if they travel frequently)
- "interest rates" (if they have debt)
- "recession 2026" (for general economic risk)
- "oil prices" (for energy costs)
- "housing market" (for rent/mortgage risk)

Return as JSON:
{{
  "vulnerabilities": ["brief description of financial vulnerability 1", "vulnerability 2"],
  "high_risk_categories": {{
    "category_name": {{"monthly_amount": number, "risk_level": "high/medium/low", "reason": "why this is risky"}}
  }},
  "market_queries": ["search query 1", "search query 2", "search query 3"],
  "summary": "2-3 sentence summary of their financial risk profile"
}}

IMPORTANT: 
- market_queries should be SHORT (2-3 words) and focused on SEARCHABLE prediction market topics
- Base queries on ACTUAL spending categories from the bank statement
- Focus on price/inflation risks, not generic advice"""
            }],
            response_format={"type": "json_object"}
        )
        
        risk_analysis = json.loads(analysis_response.choices[0].message.content)
        print(f"Risk analysis: {json.dumps(risk_analysis, indent=2)}")
        
        # Step 3: Fetch prediction markets based on spending patterns
        hedge_markets = []
        market_queries = risk_analysis.get("market_queries", [])
        
        if market_queries:
            probable_api_key = os.getenv("PROBABLE_API_KEY")
            probable_api_url = os.getenv("PROBABLE_API_URL", "https://probable-api.netlify.app/api")
            
            if probable_api_key:
                print(f"Fetching markets for queries: {market_queries}")
                async with httpx.AsyncClient(timeout=20.0) as client:
                    for query in market_queries[:5]:  # Limit to 5 queries
                        try:
                            print(f"  → Searching: '{query}'")
                            response = await client.get(
                                f"{probable_api_url}/search-markets",
                                params={
                                    "query": query,
                                    "limit": "4",  # Get top 4 for each query
                                    "includeClosed": "false"
                                },
                                headers={"x-api-key": probable_api_key}
                            )
                            if response.status_code == 200:
                                markets_data = response.json()
                                if markets_data.get("markets"):
                                    # Add query context to each market
                                    for market in markets_data["markets"][:2]:  # Take top 2 from each query
                                        market["hedge_category"] = query  # Tag with the query for context
                                        hedge_markets.append(market)
                                    print(f"    ✓ Found {len(markets_data['markets'])} markets")
                            else:
                                print(f"    ✗ API returned {response.status_code}")
                        except Exception as e:
                            print(f"    ✗ Error: {e}")
                print(f"\nTotal hedge markets found: {len(hedge_markets)}")
        
        # Combine all analysis
        full_analysis = {
            "extracted_data": extracted_data,
            "risk_analysis": risk_analysis,
            "hedge_markets": hedge_markets,  # Add actual markets
            "analyzed_at": datetime.now(UTC).isoformat(),
            "document_name": file.filename
        }
        
        # Get existing profile
        profile_result = sb.table("profiles").select("profile_json").eq(
            "user_id", user_id
        ).execute()
        
        if not profile_result.data or len(profile_result.data) == 0:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        profile_data = profile_result.data[0]
        
        # Update profile_json with financial analysis (support multiple documents)
        existing_json = profile_data.get("profile_json") or {}
        
        # Store as array to support multiple documents (max 3)
        if "financial_analyses" not in existing_json:
            existing_json["financial_analyses"] = []
        
        # Add new analysis
        existing_json["financial_analyses"].append(full_analysis)
        
        # Keep only the latest 3 analyses
        existing_json["financial_analyses"] = existing_json["financial_analyses"][-3:]
        
        # Save to profile
        sb.table("profiles").update({
            "profile_json": existing_json,
            "updated_at": datetime.now(UTC).isoformat()
        }).eq("user_id", user_id).execute()
        
        print(f"✓ Saved analysis to profile for user {user_id}")
        
        # Return user-friendly response
        return DocumentAnalysis(
            status="analyzed",
            analysis={
                "income": extracted_data.get("income"),
                "expenses": extracted_data.get("expenses"),
                "vulnerabilities": risk_analysis.get("vulnerabilities", []),
                "market_queries": market_queries,  # The search queries used
                "hedge_markets": hedge_markets,  # Actual prediction markets
                "summary": risk_analysis.get("summary", ""),
                "analyzed_at": full_analysis["analyzed_at"]
            }
        )
        
    except json.JSONDecodeError as e:
        print(f"Error parsing JSON from LLM: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse analysis results: {str(e)}"
        )
    except Exception as e:
        print(f"Error analyzing document: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing document: {str(e)}"
        )

@router.get("/{user_id}/financial-analysis")
def get_financial_analysis(user_id: str):
    """Get the stored financial analysis for a user (returns all analyses, up to 3)"""
    try:
        profile = sb.table("profiles").select("profile_json").eq(
            "user_id", user_id
        ).execute()
        
        if not profile.data or len(profile.data) == 0:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        profile_data = profile.data[0]
        profile_json = profile_data.get("profile_json") or {}
        
        # Support new multi-analysis format
        financial_analyses = profile_json.get("financial_analyses", [])
        
        # Backward compatibility: check for old single analysis format
        if not financial_analyses and profile_json.get("financial_analysis"):
            financial_analyses = [profile_json.get("financial_analysis")]
        
        if not financial_analyses:
            return {"status": "no_analysis", "message": "No financial document has been analyzed yet"}
        
        return {
            "status": "found",
            "analyses": financial_analyses,  # Return all analyses
            "count": len(financial_analyses)
        }
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error getting financial analysis: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{user_id}/financial-analysis")
def delete_financial_analysis(user_id: str, analysis_index: int = None):
    """Remove financial analysis from user profile (optionally specify which one)"""
    try:
        # Get existing profile
        profile = sb.table("profiles").select("profile_json").eq(
            "user_id", user_id
        ).execute()
        
        if not profile.data or len(profile.data) == 0:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        profile_data = profile.data[0]
        
        # Remove financial analysis
        profile_json = profile_data.get("profile_json") or {}
        
        if analysis_index is not None:
            # Delete specific analysis
            if "financial_analyses" in profile_json:
                analyses = profile_json["financial_analyses"]
                if 0 <= analysis_index < len(analyses):
                    analyses.pop(analysis_index)
                    profile_json["financial_analyses"] = analyses
        else:
            # Delete all analyses
            if "financial_analyses" in profile_json:
                del profile_json["financial_analyses"]
            if "financial_analysis" in profile_json:
                del profile_json["financial_analysis"]
        
        # Update profile
        sb.table("profiles").update({
            "profile_json": profile_json,
            "updated_at": datetime.now(UTC).isoformat()
        }).eq("user_id", user_id).execute()
        
        return {"status": "deleted", "message": "Financial analysis removed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
