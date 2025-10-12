from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from api.schemas import TransactionData, InferenceData
from api.refund_predictor import data_process


app = FastAPI(
    title="MavenRail",
    description="Predict the status of refund request based on various ticket-related factors",
    version="2.5.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/docs",
    redoc_url=None,
)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.post("/inference/", response_model=InferenceData)
async def create_transaction(transaction: TransactionData):
    try:
        result = data_process(transaction_data=transaction)
        return InferenceData(**result)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# Health check endpoint (good practice for Render)
@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    import os

    # Use PORT from environment (Render provides this)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
