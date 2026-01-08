from fastapi import FastAPI
from pydantic import BaseModel
from predict_pipeline import FeedbackAnalyzer

app = FastAPI()
analyzer = FeedbackAnalyzer()

class FeedbackInput(BaseModel):
    text: str
    user_type: str
    timestamp: str

@app.post("/analyze")
def analyze_feedback(input: FeedbackInput):
    return analyzer.analyze(input.text, input.user_type, input.timestamp)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
