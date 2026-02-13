import pandas as pd
import re
import joblib
import nltk
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from datetime import datetime
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
import requests
import json

from sklearn.ensemble import RandomForestClassifier


# Gemma constants
OLLAMA_URL = "http://localhost:11434/api/chat"
GEMMA_SYSTEM = """
You are a strict spell correction engine

Rules:
- Correct spelling mistakes only.
- Do not explain.
- Do not rephrase.
- Do not change grammar.
- Do not add words.
- Do not remove words.
- Output ONLY the corrected sentence.
If there are no spelling mistakes, return the input exactly.
"""

def correct_with_gemma(text: str) -> str:
    """Spell-correct using Ollama model gemma-spellcheck:latest. Fallback to original on error."""
    payload = {
        "model": "gemma-spellcheck:latest",
        "prompt": text,
        "stream": False,
        "options": {"temperature": 0.0}
    }

    try:
        resp = requests.post("http://localhost:11434/api/generate", json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        return data.get("response", "").strip() or text
    except Exception as e:
        print(f"⚠️  Gemma error: {e}")
        return text



def validate_with_gemma(text: str) -> str:
    payload = {
        "model": "gemma-spellcheck:latest",
        "messages": [
            {
                "role": "system",
                "content": "You classify college feedback as Valid or Invalid. Reply with only one word."
            },
            {
                "role": "user",
                "content": f"""
Classify this college feedback as Valid or Invalid.

Valid:
- Meaningful college-related feedback
- Complaint, suggestion, request, or observation

Invalid:
- Greeting
- Test message
- Random words
- Meaningless text

Feedback:
{text}
"""
            }
        ],
        "stream": False,
        "options": {"temperature": 0.0}
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        result = resp.json()["message"]["content"].strip().lower()
        return "Valid" if result == "valid" else "Invalid"
    except Exception as e:
        print(f"⚠️ Validation error: {e}")
        return "Invalid"

# Initialize lemmatizer
LEMMATIZER = WordNetLemmatizer()

# Ensure WordNet corpus is downloaded
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)


try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


STOPWORDS = nltk.corpus.stopwords.words('english')
SENTIMENT_ANALYZER = SentimentIntensityAnalyzer()


# Invalid words that should always be marked as Invalid (for Rule 5)
INVALID_PHRASES = {
    'hi', 'hello', 'hey', 'bye', 'goodbye', 'thanks', 'thank you',
    'ok', 'okay', 'sure', 'yes', 'no', 'maybe', 'perhaps',
    'just checking', 'testing', 'test', 'random', 'sample',
    'feedback', 'nothing', 'no comments', 'no major issues',
    'hello world', 'hi there', 'word', 'words',
    'checking', 'message', 'text', 'data', 'example',
    'asdf', 'qwerty', 'test message', 'sample text'
}


# College context keywords for Rule 1 (Relevance to College Context)
COLLEGE_CONTEXT_KEYWORDS = {
    'class', 'course', 'professor', 'teacher', 'exam', 'lecture', 'assignment',
    'grade', 'subject', 'syllabus', 'study', 'academic', 'classroom',
    'library', 'book', 'reading', 'research', 'manuscript', 'shelf', 'borrow',
    'canteen', 'cafe', 'food', 'meal', 'lunch', 'dinner', 'menu', 'beverage',
    'maintenance', 'repair', 'broken', 'damaged', 'fix', 'cleaning', 'facility',
    'plumbing', 'electrical', 'pipe', 'water', 'fan', 'ac', 'light', 'ceiling',
    'admin', 'office', 'documentation', 'paperwork', 'fee', 'enrollment',
    'website', 'portal', 'login', 'app', 'online', 'browser', 'server',
    'transport', 'bus', 'vehicle', 'commute'
}

class CollegeFeedbackPredictor:
    def __init__(self):
        self.models_trained = False
        self.ensure_dirs()
        self.load_or_train_models()
  
    def ensure_dirs(self):
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
  
    
    def clean_text(self, text):
        if pd.isna(text) or str(text).strip() == '':
            return 'empty'
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
        words = [LEMMATIZER.lemmatize(w) for w in words]
        return ' '.join(words) if words else 'empty'
    
    # def keyword_boost(self, text, category):
    #     """Check if text contains category keywords"""
    #     text_lower = text.lower()
    #     keywords = CATEGORY_KEYWORDS.get(category, [])
    #     count = sum(1 for kw in keywords if kw in text_lower)
    #     return count
        
    def load_data(self):
        csv_path = 'data/training_data.csv'
        if not os.path.exists(csv_path):
            print(f"❌ Create {csv_path}")
            print("Format: id,feedback_message,valid_invalid,category,sentiment")
            return None
        
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows | Columns: {df.columns.tolist()}")
        
        # Ignore category column (keep in CSV)
        df['clean_feedback'] = df['feedback_message'].apply(self.clean_text)
        df['valid_num'] = df['valid_invalid'].map({'Valid': 1, 'Invalid': 0}).fillna(0)
        
        print(f"✅ {len(df)} rows ready (sentiment only)")
        return df

    
    def load_or_train_models(self):
        if os.path.exists('models/validity_model.pkl'):
            print("⚡ Loading saved SVM models...\n")
            self._load_models()
            self.models_trained = True
            return
        
        df = self.load_data()
        if df is None:
            return
        
        self.train_all_models(df)
        self.train_category_model(df)

    
    def _load_models(self):
        # Only load what we use
        # self.cat_model = joblib.load('models/category_model.pkl')
        # self.cat_vec = joblib.load('models/category_vec.pkl')
        self.sent_model = joblib.load('models/sentiment_model.pkl')
        self.sent_vec = joblib.load('models/sentiment_vec.pkl')
        if os.path.exists("models/category_model.pkl"):
            self.cat_model = joblib.load("models/category_model.pkl")
            self.cat_vec = joblib.load("models/category_vec.pkl")


    
    def train_all_models(self, df):
        print("\n⚡ TRAINING SENTIMENT SVM:\n")
        
        # Sentiment only
        print("🔥 Sentiment SVM...", end=' ', flush=True)
        df_valid = df[df['valid_num'] == 1]
        Xs = df_valid['clean_feedback']
        ys = df_valid['sentiment']
        
        if len(ys.unique()) < 2:
            print("⚠️ Single sentiment class")
        else:
            Xs_train, Xs_test, ys_train, ys_test = train_test_split(
                Xs, ys, test_size=0.2, random_state=42, stratify=ys
            )
            
            self.sent_vec = TfidfVectorizer(max_features=800, ngram_range=(1,2))
            Xs_train_tfidf = self.sent_vec.fit_transform(Xs_train)
            Xs_test_tfidf = self.sent_vec.transform(Xs_test)
            
            self.sent_model = SVC(kernel='rbf', probability=True, random_state=42)
            self.sent_model.fit(Xs_train_tfidf, ys_train)
            print(f"✅ {self.sent_model.score(Xs_test_tfidf, ys_test):.1%}")
        
        # Save sentiment only
        joblib.dump(self.sent_model, 'models/sentiment_model.pkl')
        joblib.dump(self.sent_vec, 'models/sentiment_vec.pkl')
        self.models_trained = True
        print("🎉 Sentiment model saved!")



    



    # ================= CATEGORY RANDOM FOREST =================

    def train_category_model(self, df):
        print("\n⚡ TRAINING CATEGORY RANDOM FOREST:\n")

        df_valid = df[df['valid_num'] == 1]

        Xc = df_valid['clean_feedback']
        yc = df_valid['category']

        self.cat_vec = TfidfVectorizer(max_features=1000, ngram_range=(1,2))
        Xc_tfidf = self.cat_vec.fit_transform(Xc)

        self.cat_model = RandomForestClassifier(
            n_estimators=200,
            random_state=42
        )

        self.cat_model.fit(Xc_tfidf, yc)

        joblib.dump(self.cat_model, "models/category_model.pkl")
        joblib.dump(self.cat_vec, "models/category_vec.pkl")

        print("🎉 Category Random Forest saved!")


    def load_category_model(self):
        if os.path.exists("models/category_model.pkl"):
            self.cat_model = joblib.load("models/category_model.pkl")
            self.cat_vec = joblib.load("models/category_vec.pkl")
            return True
        return False









    def predict(self, feedback_message, user_type="student", verbose=False):
        if not self.models_trained:
            return {"error": "Models not trained"}

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        original = feedback_message

        # 1) Gemma correction
        corrected = correct_with_gemma(original)

        if verbose:
            print(f"👩‍🎓 Original: '{original}'")
            print(f"✨ Corrected: '{corrected}'")

        # 2) 7-rule validity on corrected text
        validation_result = validate_with_gemma(corrected)

        # 2) 7-rule validity on corrected text
        validation_result = validate_with_gemma(corrected)

        if validation_result == "Invalid":
            return {
                'original_text': original,
                'corrected_text': corrected,
                'valid_invalid': 'Invalid',
                'sentiment': None,
                'confidence': "0.00",
                'timestamp': ts
            }
        

        # 3) Preprocess + sentiment
        clean = self.clean_text(corrected)

        sent_tfidf = self.sent_vec.transform([clean])
        sentiment = self.sent_model.predict(sent_tfidf)[0]
        sent_prob = self.sent_model.predict_proba(sent_tfidf)[0].max()
                # Category prediction (Random Forest)
        cat_tfidf = self.cat_vec.transform([clean])
        category = self.cat_model.predict(cat_tfidf)[0]

# ADD at the end of predict() after sentiment calculation
        return {
            'original_text': original,
            'corrected_text': corrected,
            'valid_invalid': 'Valid',
            'sentiment': sentiment,
            'category': category,
            'confidence': f"{sent_prob:.2f}",
            'timestamp': ts
        }


if __name__ == "__main__":
    predictor = CollegeFeedbackPredictor()
    print("⚡ Ready! (type 'quit' to exit)\n")
    
    while True:
        text = input("👤 ").strip()
        if text.lower() in ['quit', 'exit']:
            break
        result = predictor.predict(text, verbose=True)
        if 'error' not in result:
            if result['valid_invalid'] == 'Invalid':
                print(f"📊 {result['valid_invalid']} ({result.get('criteria_passed', 'N/A')})\n")
            else:
                print(f"📊 {result['valid_invalid']} | Sentiment: {result['sentiment']} | Category: {result['category']} | Confidence: {result['confidence']}\n")

                