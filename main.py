from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

from api.schemas import TransactionData, InferenceData
from api.refund_predictor import data_process

# Mangum only needed for Lambda
try:
    from mangum import Mangum
except ImportError:
    Mangum = None  # Safe fallback for local builds

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


# Create Mangum handler only if Mangum is available
if Mangum:
    handler = Mangum(app)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
