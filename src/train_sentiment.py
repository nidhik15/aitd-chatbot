# Same pattern
df_valid = df[df['valid_invalid'] == 1]
X = df_valid['clean_feedback']
y = df_valid['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vec = get_vectorizer()
X_train_tfidf = vec.fit_transform(X_train)
X_test_tfidf = vec.transform(X_test)

model = SVC(kernel='linear', C=1.0, class_weight='balanced', probability=True, random_state=42)
model.fit(X_train_tfidf, y_train)

print(classification_report(y_test, model.predict(X_test_tfidf)))
joblib.dump(model, '../models/sentiment_svm.pkl')
joblib.dump(vec, '../models/sentiment_vec.pkl')
