import os
from typing import Optional
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

app = FastAPI(
    title="Spain Financial Platform CX Evaluation API",
    version="1.0.0",
    description="Backend service evaluating mystery shopper logs for usability and compliance."
)

class EvaluationRequest(BaseModel):
    shopper_report: str
    user_location: Optional[str] = "Spain"

class EvaluationResponse(BaseModel):
    status: str
    retrieved_rules: list[str]
    audit_summary: str

@app.get("/health")
def health_check():
    return {"status": "ok", "service": "CX Evaluation Engine"}

@app.post("/evaluate", response_model=EvaluationResponse)
def evaluate_cx(
    payload: EvaluationRequest, 
    x_openai_key: Optional[str] = Header(None, alias="X-OpenAI-Key")
):
    """FastAPI endpoint utilizing lazy imports to reduce cold-start latency."""
    api_key = x_openai_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=401, detail="OpenAI API Key required via header 'X-OpenAI-Key' or server environment.")
    
    # Lazy Import of model pipeline to accelerate server initial startup
    from model import run_cx_audit
    
    try:
        result = run_cx_audit(payload.shopper_report, api_key)
        return EvaluationResponse(
            status="success",
            retrieved_rules=result["retrieved_compliance_rules"],
            audit_summary=result["final_evaluation"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))