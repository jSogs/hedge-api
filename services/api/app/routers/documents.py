from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Dict, Any, List
from datetime import datetime, UTC
import json
import base64
import os
import io

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
        
        # Step 2: Analyze vulnerabilities and risks
        analysis_response = oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": f"""Based on this user's financial data:
{json.dumps(extracted_data, indent=2)}

Analyze:
1. What economic events or trends most affect this person's finances?
2. What are their biggest financial vulnerabilities?
3. What categories have high volatility or risk?
4. What should they consider hedging against?

Consider factors like:
- High percentage of income on variable costs (gas, food)
- Exposure to inflation, interest rates, energy prices
- Large fixed costs that could change (rent, insurance)

Return as JSON:
{{
  "vulnerabilities": ["specific vulnerability 1", "specific vulnerability 2"],
  "high_risk_categories": {{
    "category": {{"monthly_amount": number, "risk_level": "high/medium/low", "reason": "why"}}
  }},
  "hedge_suggestions": ["what to hedge against 1", "what to hedge against 2"],
  "summary": "2-3 sentence summary of their financial risk profile"
}}"""
            }],
            response_format={"type": "json_object"}
        )
        
        risk_analysis = json.loads(analysis_response.choices[0].message.content)
        print(f"Risk analysis: {json.dumps(risk_analysis, indent=2)}")
        
        # Combine all analysis
        full_analysis = {
            "extracted_data": extracted_data,
            "risk_analysis": risk_analysis,
            "analyzed_at": datetime.now(UTC).isoformat(),
            "document_name": file.filename
        }
        
        # Get existing profile
        profile_result = sb.table("profiles").select("profile_json").eq(
            "user_id", user_id
        ).single().execute()
        
        if not profile_result.data:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Update profile_json with financial analysis
        existing_json = profile_result.data.get("profile_json") or {}
        existing_json["financial_analysis"] = full_analysis
        
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
                "hedge_suggestions": risk_analysis.get("hedge_suggestions", []),
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
    """Get the stored financial analysis for a user"""
    try:
        profile = sb.table("profiles").select("profile_json").eq(
            "user_id", user_id
        ).single().execute()
        
        if not profile.data:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        profile_json = profile.data.get("profile_json") or {}
        financial_analysis = profile_json.get("financial_analysis")
        
        if not financial_analysis:
            return {"status": "no_analysis", "message": "No financial document has been analyzed yet"}
        
        return {
            "status": "found",
            "analysis": financial_analysis
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{user_id}/financial-analysis")
def delete_financial_analysis(user_id: str):
    """Remove financial analysis from user profile"""
    try:
        # Get existing profile
        profile = sb.table("profiles").select("profile_json").eq(
            "user_id", user_id
        ).single().execute()
        
        if not profile.data:
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Remove financial analysis
        profile_json = profile.data.get("profile_json") or {}
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
