from fastapi import FastAPI
from pydantic import BaseModel
from feedback_predictor import CollegeFeedbackPredictor

app = FastAPI()
predictor = CollegeFeedbackPredictor()

class FeedbackRequest(BaseModel):
    feedback_message: str
    user_type: str = "student"

@app.post("/predict")
async def predict_feedback(req: FeedbackRequest):
    return predictor.predict(req.feedback_message, req.user_type)

@app.get("/")
async def root():
    return {"message": "College Feedback SVM API - Ready!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
