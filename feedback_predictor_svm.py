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


# Category-specific keywords (7 categories only)
CATEGORY_KEYWORDS = {
    'Academics': ['academics', 'course', 'class', 'professor', 'teacher', 'study', 'exam', 'lecture', 'assignment', 'grade', 'subject', 'syllabus'],
    'Library': ['library', 'book', 'reading', 'research', 'manuscript', 'shelf', 'borrow', 'reference', 'archive'],
    'Canteen': ['canteen', 'food', 'meal', 'lunch', 'dinner', 'cook', 'taste', 'menu', 'cafe', 'snack', 'beverage', 'coffee'],
    'Maintenance': ['maintenance', 'repair', 'broken', 'damaged', 'fix', 'cleaning', 'sanitation', 'plumbing', 'electrical', 'facilities', 'fan', 'ac', 'light', 'ceiling', 'pipe', 'water'],
    'Administration': ['administration', 'admin', 'office', 'documentation', 'paperwork', 'fee', 'register', 'enrollment', 'admission', 'form'],
    'Website': ['website', 'web', 'login', 'portal', 'page', 'link', 'app', 'online', 'browser', 'server', 'platform', 'dashboard'],
    'Uncategorized': []
}


class CollegeFeedbackPredictor:
    def __init__(self):
        self.models_trained = False
        self.ensure_dirs()
        self.load_or_train_models()
  
    def ensure_dirs(self):
        os.makedirs('models', exist_ok=True)
        os.makedirs('data', exist_ok=True)
  
    # 7-CRITERIA VALIDATION KEYWORDS
    ALL_CONTEXT_WORDS = set(COLLEGE_CONTEXT_KEYWORDS)
    
    def criterion_1_relevance_to_college_context(self, text):
        text_lower = text.lower()
        words = text_lower.split()
        return any(w in self.ALL_CONTEXT_WORDS for w in words)


    
    def criterion_2_completeness_of_message(self, text):
        """Rule 2: Verify message forms complete & meaningful sentence"""
        try:
            tokens = word_tokenize(text.lower())
            pos_tags = pos_tag(tokens)
            has_noun = any(tag.startswith('NN') for _, tag in pos_tags)
            has_verb = any(tag.startswith('VB') for _, tag in pos_tags)
            completeness = has_noun and has_verb
        except:
            # Fallback: check for at least 2 words
            words = text.split()
            completeness = len(words) >= 2
        return completeness
    
    def criterion_3_presence_of_informative_keywords(self, text):
        """Rule 3: Verify if message contains domain-relevant informative words"""
        text_lower = text.lower()
        words = text_lower.split()
        stop_words = set(STOPWORDS)
        meaningful_words = [w for w in words if w not in stop_words and len(w) > 2]
        has_informative = len(meaningful_words) >= 1
        return has_informative
    
    def criterion_4_message_length_word_count(self, text):
        """Rule 4: Filter out very short or meaningless messages"""
        words = text.split()
        word_count = len(words)
        sufficient_length = word_count >= 2  # At least 4 words
        return sufficient_length
    
    def criterion_5_avoidance_of_placeholder_test_phrases(self, text):
        """Rule 5: Detect dummy/testing messages not genuine feedback"""
        text_lower = text.lower().strip()
        
        # Check exact match
        if text_lower in INVALID_PHRASES:
            return False
        
        # Check if all words are test phrases
        words = text_lower.split()
        if len(words) > 0 and all(w in INVALID_PHRASES for w in words if len(w) > 1):
            return False
        
        # Check for repeating chars (-----, =====)
        if re.match(r'^[\-\=\*_]+$', text_lower):
            return False
        
        return True
    
    def criterion_6_tone_intent_consistency(self, text):
        """Rule 6: Ensure message expresses opinion, request, or observation"""
        text_lower = text.lower()
        
        # Check for sentiment words
        negative_words = {'not', 'bad', 'terrible', 'broken', 'damaged', 'slow', 'buggy', 
                         'issue', 'problem', 'poor', 'horrible', 'awful', 'worst', 'never'}
        positive_words = {'great', 'excellent', 'good', 'amazing', 'perfect', 'nice', 
                         'wonderful', 'fantastic', 'best', 'awesome', 'love'}
        
        has_negative = any(w in text_lower for w in negative_words)
        has_positive = any(w in text_lower for w in positive_words)
        
        # Check with VADER sentiment analyzer
        try:
            sentiment_scores = SENTIMENT_ANALYZER.polarity_scores(text)
            compound = sentiment_scores['compound']
            has_sentiment = compound != 0.0  # Has some sentiment
        except:
            has_sentiment = has_negative or has_positive
        
        return has_negative or has_positive or has_sentiment
    
    def criterion_7_grammatical_contextual_coherence(self, text):
        """Rule 7: Ensure message is grammatically meaningful"""
        words = text.split()
        
        # Check minimum word properties
        meaningful_structure = True
        
        # No gibberish (very short random chars)
        if all(len(w) <= 1 for w in words if w.isalpha()):
            meaningful_structure = False
        
        # Has at least one proper word (more than 1 char)
        has_proper_word = any(len(w) > 1 for w in words if w.isalpha())
        meaningful_structure = meaningful_structure and has_proper_word
        
        return meaningful_structure
    
    def is_valid_feedback_advanced(self, text):
        """
        7-CRITERIA VALIDATION (Enhanced Version)
        Returns (is_valid, criteria_passed, total_criteria, reason)
        """
        if pd.isna(text) or str(text).strip() == '':
            return False, 0, 7, "Empty text"
        
        text = str(text).strip()
        
        # Apply all 7 criteria
        c1 = self.criterion_1_relevance_to_college_context(text)
        c2 = self.criterion_2_completeness_of_message(text)
        c3 = self.criterion_3_presence_of_informative_keywords(text)
        c4 = self.criterion_4_message_length_word_count(text)
        c5 = self.criterion_5_avoidance_of_placeholder_test_phrases(text)
        c6 = self.criterion_6_tone_intent_consistency(text)
        c7 = self.criterion_7_grammatical_contextual_coherence(text)
        
        criteria = [c1, c2, c3, c4, c5, c6, c7]
        passed = sum(criteria)
        
        # Need at least 5/7 criteria to be valid
        is_valid = passed >= 5
        
        reason = f"Criteria: C1={c1} C2={c2} C3={c3} C4={c4} C5={c5} C6={c6} C7={c7} (Passed {passed}/7)"
        
        return is_valid, passed, 7, reason
    
    # =========================================================================
    
    def clean_text(self, text):
        if pd.isna(text) or str(text).strip() == '':
            return 'empty'
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        words = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
        words = [LEMMATIZER.lemmatize(w) for w in words]
        return ' '.join(words) if words else 'empty'
    
    def keyword_boost(self, text, category):
        """Check if text contains category keywords"""
        text_lower = text.lower()
        keywords = CATEGORY_KEYWORDS.get(category, [])
        count = sum(1 for kw in keywords if kw in text_lower)
        return count
    
    def load_data(self):
        csv_path = 'data/training_data.csv'
        if not os.path.exists(csv_path):
            print(f"❌ Create {csv_path}")
            print("Format: id,feedback_message,valid_invalid,category,sentiment")
            print("Categories: Academics, Library, Canteen, Maintenance, Administration, Website, Uncategorized")
            return None
        
        df = pd.read_csv(csv_path)
        print(f"✅ Loaded {len(df)} rows")
        
        # Filter to only valid categories
        valid_cats = set(CATEGORY_KEYWORDS.keys())
        df_filtered = df[df['category'].isin(valid_cats) | (df['valid_invalid'] == 'Invalid')]
        
        if len(df_filtered) < len(df):
            print(f"⚠️  Filtered out {len(df) - len(df_filtered)} rows with invalid categories")
        
        df = df_filtered
        df['clean_feedback'] = df['feedback_message'].apply(self.clean_text)
        df['valid_num'] = df['valid_invalid'].map({'Valid': 1, 'Invalid': 0}).fillna(0)
        
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
    
    def _load_models(self):
        # Only load what we use
        self.cat_model = joblib.load('models/category_model.pkl')
        self.cat_vec = joblib.load('models/category_vec.pkl')
        self.sent_model = joblib.load('models/sentiment_model.pkl')
        self.sent_vec = joblib.load('models/sentiment_vec.pkl')

    
    def train_all_models(self, df):
        print("\n⚡ TRAINING (Category + Sentiment SVMs):\n")
        
        # 1️⃣ CATEGORY SVM (skip validity training)
        print("  2️⃣  Category SVM (7 classes, balanced)...", end=' ', flush=True)
        df_valid = df[df['valid_num'] == 1].copy()

        # Exclude Uncategorized when computing min_count
        counts = df_valid[df_valid['category'] != 'Uncategorized']['category'].value_counts()
        if len(counts) == 0:
            print("❌ No valid data for categories")
            return
            
        min_count = counts.min()

        balanced_frames = []
        for cat, cnt in counts.items():
            cat_df = df_valid[df_valid['category'] == cat]
            if cnt > min_count:
                cat_df = cat_df.sample(min_count, random_state=42)
            balanced_frames.append(cat_df)

        df_balanced = pd.concat(balanced_frames, ignore_index=True)
        Xc = df_balanced['clean_feedback']
        yc = df_balanced['category']

        print(f"\n  After balancing: {yc.value_counts().to_dict()}\n  ", end='')

        if len(yc.unique()) < 2:
            print("⚠️  Only 1 category")
        else:
            Xc_train, Xc_test, yc_train, yc_test = train_test_split(
                Xc, yc, test_size=0.2, random_state=42, stratify=yc
            )
            
            self.cat_vec = TfidfVectorizer(
                max_features=1200,
                ngram_range=(1,3),
                min_df=1,
                max_df=0.9,
                sublinear_tf=True
            )
            Xc_train_tfidf = self.cat_vec.fit_transform(Xc_train)
            Xc_test_tfidf = self.cat_vec.transform(Xc_test)
            
            self.cat_model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                decision_function_shape='ovr',
                random_state=42,
            )
            self.cat_model.fit(Xc_train_tfidf, yc_train)
            print(f"✅ {self.cat_model.score(Xc_test_tfidf, yc_test):.1%}")
        
        # 2️⃣ SENTIMENT SVM
        print("  3️⃣  Sentiment SVM...", end=' ', flush=True)
        Xs = df_valid['clean_feedback']
        ys = df_valid['sentiment']
        
        if len(ys.unique()) < 2:
            print("⚠️  Single sentiment class")
        else:
            Xs_train, Xs_test, ys_train, ys_test = train_test_split(
                Xs, ys, test_size=0.2, random_state=42, stratify=ys
            )
            
            self.sent_vec = TfidfVectorizer(
                max_features=800, 
                ngram_range=(1,2), 
                min_df=1, 
                max_df=0.8
            )
            Xs_train_tfidf = self.sent_vec.fit_transform(Xs_train)
            Xs_test_tfidf = self.sent_vec.transform(Xs_test)
            
            self.sent_model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,
                decision_function_shape='ovr',
                random_state=42,
            )
            self.sent_model.fit(Xs_train_tfidf, ys_train)
            print(f"✅ {self.sent_model.score(Xs_test_tfidf, ys_test):.1%}")
        
        # SAVE (only category + sentiment models)
        joblib.dump(self.cat_model, 'models/category_model.pkl')
        joblib.dump(self.cat_vec, 'models/category_vec.pkl')
        joblib.dump(self.sent_model, 'models/sentiment_model.pkl')
        joblib.dump(self.sent_vec, 'models/sentiment_vec.pkl')
        
        self.models_trained = True
        print("\n🎉 Category + Sentiment models saved!\n")

    
    def predict(self, feedback_message, user_type="student", verbose=False):
        if not self.models_trained:
            return {"error": "Models not trained"}
        
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # LOCATION 1️⃣ - 7-CRITERIA VALIDATION (ENHANCED)
        is_valid, criteria_passed, total_criteria, criteria_reason = self.is_valid_feedback_advanced(feedback_message)
        
        if not is_valid:
            if verbose:
                print(f"📋 {criteria_reason}")
            return {
                'feedback_message': feedback_message,
                'valid_invalid': 'Invalid',
                'category': None,
                'sentiment': None,
                'confidence': f"0.{10-criteria_passed}0",
                'criteria_passed': f"{criteria_passed}/{total_criteria}",
                'timestamp': ts
            }
        
        clean = self.clean_text(feedback_message)
        
        cat_tfidf = self.cat_vec.transform([clean])
        cat_pred_probs = self.cat_model.predict_proba(cat_tfidf)[0]
        
        boosted_scores = []
        for i, category in enumerate(self.cat_model.classes_):
            base_score = cat_pred_probs[i]
            keyword_count = self.keyword_boost(feedback_message, category)
            boosted_score = base_score + (keyword_count * 0.15)
            boosted_scores.append(boosted_score)
        
        best_idx = boosted_scores.index(max(boosted_scores))
        category = self.cat_model.classes_[best_idx]
        
        # 3. SENTIMENT
        sent_tfidf = self.sent_vec.transform([clean])
        sentiment = self.sent_model.predict(sent_tfidf)[0]
        sent_prob = self.sent_model.predict_proba(sent_tfidf)[0].max()
        
        # LOCATION 4️⃣ - FINAL RETURN (VALID FEEDBACK)
        return {
            'feedback_message': feedback_message,
            'valid_invalid': 'Valid',
            'category': category,
            'sentiment': sentiment,
            'confidence': f"{sent_prob:.2f}",
            'criteria_passed': f"{criteria_passed}/{total_criteria}",
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
                print(f"📊 {result['valid_invalid']} | {result['category']} | {result['sentiment']} | Criteria: {result.get('criteria_passed', 'N/A')}\n")