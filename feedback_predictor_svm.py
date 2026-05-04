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
    payload = {
        "model": "gemma-spellcheck:latest",
        "messages": [
            {
                "role": "system",
                "content": """
You are an intelligent text normalization engine.

Your task:
1) Correct spelling mistakes.
2) Replace slang words with proper English meaning.
3) Expand informal abbreviations.
4) Replace emojis with contextually correct English words that fit naturally into the sentence.

Emoji Handling Rules:
- Do NOT append emotion words at the end.
- Replace the emoji with a grammatically suitable word.
- The final sentence must sound natural.
- Do NOT create phrases like "bad angry".
- If the emoji expresses sentiment, convert it into a suitable adjective.

Examples:

"wifi speed is 👎" → "wifi speed is bad"
"canteen food is 🤢" → "canteen food is disgusting"
"class was 🔥" → "class was excellent"
"admin response 🤬" → "admin response is terrible"
"library staff 😊" → "library staff are friendly"
"exam schedule 😡" → "exam schedule is frustrating"
"wifi slow 😭" → "wifi is very slow"

Rules:
- Do NOT rephrase entire sentence.
- Do NOT change sentence meaning.
- Do NOT add new ideas.
- Keep college-related words unchanged
  (canteen, library, admin, portal, syllabus, faculty, hostel, etc.)
- Output ONLY the corrected sentence.



If no correction is needed, return the input exactly.
"""
            },
            {
                "role": "user",
                "content": text
            }
        ],
        "stream": False,
        "options": {
            "temperature": 0.0,
            "top_p": 0.0
        }
    }

    try:
        resp = requests.post(OLLAMA_URL, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ Gemma error: {e}")
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
- Any feedback related to college life
- Includes food, facilities, staff, academics, or services
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


# Get stopwords
STOPWORDS = set(stopwords.words('english'))

# Keep important sentiment words
NEGATION_WORDS = {'not', 'no', 'nor', 'too', 'very'}

# Remove them from stopwords
STOPWORDS = STOPWORDS - NEGATION_WORDS

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

CATEGORY_KEYWORDS = {
    "academics": {
        'class', 'course', 'professor', 'teacher', 'exam', 'lecture', 'assignment',
        'grade', 'subject', 'syllabus', 'study', 'academic', 'classroom'
    },

    "library": {
        'library', 'book', 'reading', 'research', 'manuscript', 'shelf', 'borrow'
    },

    "canteen": {
        'canteen', 'cafe', 'food', 'meal', 'lunch', 'dinner', 'menu', 'beverage'
    },

    "facilities": {
        'maintenance', 'repair', 'broken', 'damaged', 'fix', 'cleaning', 'facility',
        'plumbing', 'electrical', 'pipe', 'water', 'fan', 'ac', 'light', 'ceiling',
        'transport', 'bus', 'vehicle', 'commute'
    },

    "administration": {
        'admin', 'office', 'documentation', 'paperwork', 'fee', 'enrollment', 'staff', 'management'
    },

    "website": {
        'website', 'portal', 'login', 'app', 'online', 'browser', 'server'
    }
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
        words = text.split()

        cleaned_words = []
        for w in words:
            if w not in STOPWORDS:
                lemma = LEMMATIZER.lemmatize(w)
                cleaned_words.append(lemma)

        return ' '.join(cleaned_words) if cleaned_words else 'empty'
    
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
        # Normalize case first
        df['valid_invalid'] = df['valid_invalid'].str.strip().str.lower()
        df['sentiment'] = df['sentiment'].str.strip().str.lower()

        df['clean_feedback'] = df['feedback_message'].apply(self.clean_text)
        # 🔥 Save preprocessed dataset
        df_clean = df[['clean_feedback', 'valid_invalid', 'category', 'sentiment']]

        df_clean.to_csv('data/preprocessed_training_data.csv', index=False)

        print("💾 Clean-only dataset saved!")
        print("💾 Preprocessed dataset saved as data/preprocessed_training_data.csv")

        # Map lowercase values
        df['valid_num'] = df['valid_invalid'].map({'valid': 1, 'invalid': 0}).fillna(0)


        
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
            
            self.sent_vec = TfidfVectorizer(
                max_features=2000,
                ngram_range=(1,3),   # VERY IMPORTANT
                min_df=2
            )
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


    def predict(self, feedback_message, user_type="student", verbose=False):
        if not self.models_trained:
            return {"error": "Models not trained"}

        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        original = feedback_message

        # 🔥 Step 0: Empty input check
        if original is None or str(original).strip() == "":
            return {
                'original_text': original,
                'corrected_text': '',
                'valid_invalid': 'Invalid',
                'results': [],
                'message': 'Empty input not allowed',
                'timestamp': ts
            }

        # 🔥 Step 1: Gemma normalization FIRST
        corrected = correct_with_gemma(original)

        if verbose:
            print(f"👩‍🎓 Original: '{original}'")
            print(f"✨ Corrected: '{corrected}'")

        # 🔥 Step 2: Rule-based validation on CORRECTED text
        if not self.rule_based_validation(corrected):
            return {
                'original_text': original,
                'corrected_text': corrected,
                'valid_invalid': 'Invalid',
                'results': [],
                'message': 'Failed rule-based validation',
                'timestamp': ts
            }

        # 🔥 Step 3: LLM validation
        validation_result = validate_with_gemma(corrected)

        if validation_result == "Invalid":
            return {
                'original_text': original,
                'corrected_text': corrected,
                'valid_invalid': 'Invalid',
                'results': [],
                'timestamp': ts
            }

        # 🔥 Step 4: Sentence processing
        sentences = self.split_into_sentences(corrected)

        results = []

        for sentence in sentences:
            clean_basic = self.clean_text(sentence)
            clean = sentence.lower() + " " + clean_basic

            if clean == "empty":
                continue

            # Sentiment prediction
            sent_tfidf = self.sent_vec.transform([clean])
            sentiment = self.sent_model.predict(sent_tfidf)[0]

            # Negation handling
            sentiment = self.handle_negation_cases(sentence, sentiment)
            sent_prob = self.sent_model.predict_proba(sent_tfidf)[0].max()

            # Category detection
            category = self.rule_based_category(sentence)

            if category is None:
                cat_tfidf = self.cat_vec.transform([clean])
                category = self.cat_model.predict(cat_tfidf)[0]

            results.append({
                "sentence": sentence,
                "category": category,
                "sentiment": sentiment,
                "confidence": f"{sent_prob:.2f}"
            })

        # 🔥 Step 5: Final safeguard
        if not results:
            return {
                'original_text': original,
                'corrected_text': corrected,
                'valid_invalid': 'Invalid',
                'results': [],
                'message': 'No meaningful sentences found',
                'timestamp': ts
            }

        return {
            'original_text': original,
            'corrected_text': corrected,
            'valid_invalid': 'Valid',
            'results': results,
            'timestamp': ts
        }

    def load_category_model(self):
        if os.path.exists("models/category_model.pkl"):
            self.cat_model = joblib.load("models/category_model.pkl")
            self.cat_vec = joblib.load("models/category_vec.pkl")
            return True
        return False



    def split_into_sentences(self, text):
        text = text.strip()

        # Step 1: Split by punctuation first
        base_sentences = re.split(r'[.?!]+', text)

        final_sentences = []

        # 🔹 Strong polarity-shift conjunctions (always split)
        contrast_words = [
            "but", "however", "although", "though",
            "even though", "yet", "whereas", "while",
            "nevertheless", "nonetheless", "still",
            "on the other hand"
        ]

        # 🔹 Cause / effect (split carefully)
        cause_words = [
            "because", "since", "therefore",
            "thus", "hence", "as a result"
        ]

        # 🔹 Addition (split only in longer sentences)
        addition_words = [
            "and", "also", "moreover",
            "furthermore", "in addition"
        ]

        # Combine strong split words
        strong_pattern = r',?\s*\b(?:' + '|'.join(contrast_words + cause_words) + r')\b\s*'

        addition_pattern = r',?\s*\b(?:' + '|'.join(addition_words) + r')\b\s*'

        for sentence in base_sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # First split on strong polarity shifts
            parts = re.split(strong_pattern, sentence, flags=re.IGNORECASE)

            temp_parts = []

            for part in parts:
                part = part.strip()
                if not part:
                    continue

                # Split on addition words only if sentence is long
                if len(part.split()) > 8:
                    subparts = re.split(addition_pattern, part, flags=re.IGNORECASE)
                    temp_parts.extend(subparts)
                else:
                    temp_parts.append(part)

            for final in temp_parts:
                final = final.strip()
                if len(final.split()) >= 3:
                    final_sentences.append(final)

        return final_sentences

    def handle_negation_cases(self, sentence, predicted_sentiment):
        text = sentence.lower()
        words = text.split()

        # 🔥 Request/complaint detection
        request_words = ["fix", "improve", "repair", "resolve", "update", "check", "look into"]
        if any(word in text for word in request_words):
            return "negative"

        # ✅ Step 1: Handle SPECIAL phrases FIRST
        positive_phrases = [
            "not bad", "not worst", "not poor",
            "not terrible", "not awful", "not disappointing"
        ]

        negative_phrases = [
            "not good", "not nice", "not great",
            "not fresh", "not clean", "not working"
        ]

        for p in positive_phrases:
            if p in text:
                return "positive"

        for p in negative_phrases:
            if p in text:
                return "negative"

        # 🔥 NEW FIX: handle "not very clean", "not really good", etc.
        positive_words = [
            "good", "great", "excellent", "clean",
            "fast", "helpful", "delicious",
            "tasty", "fresh", "amazing"
        ]

        negative_words = [
            "bad", "worst", "poor", "slow", "issue",
            "problem", "delay", "dirty", "crowded",
            "unhygienic", "disgusting", "tasteless",
            "spoiled", "rotten", "awful"
        ]

        for i, word in enumerate(words):
            if word == "not":
                window = words[i:i+3]   # ['not', 'very', 'clean']

                for w in window:
                    if w in positive_words:
                        return "negative"
                    if w in negative_words:
                        return "positive"

        # ✅ Step 2: normal detection (FIXED → use words, not text)
        if any(w in words for w in negative_words):
            return "negative"

        if any(w in words for w in positive_words):
            return "positive"

        return predicted_sentiment
 
    def rule_based_validation(self, text):
        text_lower = text.lower().strip()

        # Rule 2 + 4: Length check
        if len(text_lower.split()) < 3:
            return False

        # Rule 5: Invalid phrases
        if text_lower in INVALID_PHRASES:
            return False

        # Rule 3: Keyword presence
        keyword_match = any(word in text_lower for word in COLLEGE_CONTEXT_KEYWORDS)

        if not keyword_match:
            # allow if sentence has meaningful structure
            if len(text_lower.split()) >= 5:
                return True
            return False

        return True


    def rule_based_category(self, sentence):
        words = set(sentence.lower().split())

        category_scores = {}

        # Count keyword matches for each category
        for category, keywords in CATEGORY_KEYWORDS.items():
            match_count = len(words & keywords)
            if match_count > 0:
                category_scores[category] = match_count

        # Return category with highest matches
        if category_scores:
            return max(category_scores, key=category_scores.get)

        return None



if __name__ == "__main__":
    predictor = CollegeFeedbackPredictor()

    while True:
        user_input = input("\nEnter feedback (or type 'exit'): ")

        if user_input.lower() == "exit":
            break

        result = predictor.predict(user_input, verbose=True)

        if result['valid_invalid'] == 'Invalid':
            print("\n📊 INVALID FEEDBACK\n")
        else:
            print("\n📊 VALID FEEDBACK\n")
            for r in result['results']:
                print(f"➡ Sentence: {r['sentence']}")
                print(f"   Category: {r['category']}")
                print(f"   Sentiment: {r['sentiment']}")
                print(f"   Confidence: {r['confidence']}\n")