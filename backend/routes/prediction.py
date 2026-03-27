from fastapi import APIRouter, HTTPException
from models.schemas import PredictionInput, PredictionOutput
from services.dropout_service import dropout_service

router = APIRouter()

@router.post("/predict", response_model=PredictionOutput)
async def predict_dropout(student_data: PredictionInput):
    """
    Predicts the dropout risk of a student based on their engagement features.
    """
    try:
        # Perform prediction using the service
        prediction = dropout_service.predict(student_data.dict())
        
        return PredictionOutput(
            risk=prediction["risk"],
            probability=prediction["probability"]
        )
        
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
