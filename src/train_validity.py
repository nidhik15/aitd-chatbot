import pandas as pd
import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from preprocess import load_data, get_vectorizer

df = load_data('../data/training_data.csv')
X = df['clean_feedback']
y = df['valid_invalid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vec = get_vectorizer()
X_train_tfidf = vec.fit_transform(X_train)
X_test_tfidf = vec.transform(X_test)

model = SVC(kernel='rbf', C=1.5, gamma='scale', class_weight='balanced', probability=True, random_state=42)
model.fit(X_train_tfidf, y_train)

preds = model.predict(X_test_tfidf)
print("Validity Accuracy:", accuracy_score(y_test, preds))
print(classification_report(y_test, preds))
joblib.dump(model, '../models/validity_svm.pkl')
joblib.dump(vec, '../models/validity_vec.pkl')
