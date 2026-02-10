# College Feedback SVM Pipeline

Sequential SVM analysis for Smart College Web Portal:
1. Validity Check (Valid/Invalid)
2. Category Classification (11 classes from your data)
3. Sentiment Analysis (Positive/Neutral/Negative)

## Quick Start
```bash
pip install -r requirements.txt
# Add your data to data/training_data.csv
python src/train_validity.py
python src/train_category.py  
python src/train_sentiment.py
python src/predict_pipeline.py  # Test
uvicorn src.api:app --reload
