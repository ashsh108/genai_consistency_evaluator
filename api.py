from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from core.evaluator import ConsistencyValidator

app = FastAPI(title="GenAI Consistency Engine")
validator_instance = None

class EvalRequest(BaseModel):
    source_context: str
    generated_text: str

@app.on_event("startup")
def load_models():
    global validator_instance
    validator_instance = ConsistencyValidator()

@app.post("/v1/evaluate")
def run_evaluation(payload: EvalRequest):
    if not payload.source_context or not payload.generated_text:
        raise HTTPException(status_code=400, detail="Missing context or generated text")
    
    try:
        results = validator_instance.analyze(payload.source_context, payload.generated_text)
        human_summary = synthesize_diagnostic_report(results)
        
        return_results =  {
            "status": "success", 
            "summary": human_summary,
            "data": results
        }
        return {"status": "success", "data": return_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
