# Same imports/setup
df_valid = df[df['valid_invalid'] == 1]  # Only valid samples
X = df_valid['clean_feedback']
y = df_valid['category']

# Handle small dataset with balanced weights
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

vec = get_vectorizer()
X_train_tfidf = vec.fit_transform(X_train)
X_test_tfidf = vec.transform(X_test)

model = SVC(kernel='rbf', C=2.0, gamma='scale', class_weight='balanced', decision_function_shape='ovr', random_state=42)
model.fit(X_train_tfidf, y_train)

print(classification_report(y_test, model.predict(X_test_tfidf)))
joblib.dump(model, '../models/category_svm.pkl')
joblib.dump(vec, '../models/category_vec.pkl')
