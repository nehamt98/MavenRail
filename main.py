from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
import uvicorn

from src.schemas import TransactionData, InferenceData
from src.refund_predictor import data_process

app = FastAPI(
    title="MavenRail",
    description="Predict the status of refund request based on based on various ticket-related factors",
    version="2.5.0",
    openapi_url="/api/v1/openapi.json",
    docs_url="/docs",
    redoc_url=None,  # Disable ReDoc
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
