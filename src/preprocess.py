# ... (same as before)
def load_data(file_path):
    df = pd.read_csv(file_path)
    df['clean_feedback'] = df['feedback_message'].fillna('').astype(str).apply(clean_text)
    df['valid_invalid'] = df['valid_invalid'].map({'Valid': 1, 'Invalid': 0})
    return df
