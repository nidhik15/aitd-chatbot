import joblib
import pandas as pd
from datetime import datetime
from preprocess import clean_text

class CollegeFeedbackAnalyzer:
    def __init__(self):
        self.valid_model = joblib.load('models/validity_svm.pkl')
        self.valid_vec = joblib.load('models/validity_vec.pkl')
        # self.cat_model = joblib.load('models/category_svm.pkl')
        # self.cat_vec = joblib.load('models/category_vec.pkl')
        self.sent_model = joblib.load('models/sentiment_svm.pkl')
        self.sent_vec = joblib.load('models/sentiment_vec.pkl')

    def analyze_feedback(self, feedback_message, user_type="student"):
        clean = clean_text(str(feedback_message))
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # 1. Validity Check
        valid_vec = self.valid_vec.transform([clean])
        valid_pred = self.valid_model.predict(valid_vec)[0]
        valid_prob = self.valid_model.predict_proba(valid_vec)[0][1]  # Prob of Valid
        
        if valid_pred == 0 or valid_prob < 0.6:
            return {
                'id': None,
                'feedback_message': feedback_message,
                'valid_invalid': 'Invalid',
                # 'category': 'Uncategorized',
                'sentiment': 'Neutral',
                'user_type': user_type,
                'timestamp': timestamp,
                'reason': 'Low relevance/confidence'
            }
        
        # 2. Category Classification
        # cat_vec = self.cat_vec.transform([clean])
        # category = self.cat_model.predict(cat_vec)[0]
        
        # 3. Sentiment Analysis
        sent_vec = self.sent_vec.transform([clean])
        sentiment = self.sent_model.predict(sent_vec)[0]
        
        result = {
            'id': pd.Timestamp.now().timestamp(),  # Auto ID
            'feedback_message': feedback_message,
            'valid_invalid': 'Valid',
            # 'category': category,
            'sentiment': sentiment,
            'user_type': user_type,
            'timestamp': timestamp
        }
        
        # Append to CSV
        pd.DataFrame([result]).to_csv('../data/analyzed_feedback.csv', mode='a', header=False, index=False)
        return result

# Test with your data style
analyzer = CollegeFeedbackAnalyzer()
test_feedback = "website needs improvement"
print(analyzer.analyze_feedback(test_feedback))
