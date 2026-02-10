import pandas as pd

df = pd.read_csv('data/training_data.csv')
print(f"✅ Loaded {len(df)} rows")
print("\nvalid_invalid counts:")
print(df['valid_invalid'].value_counts())
print("\nCategories:", df['category'].nunique(), "unique")
print("Sentiments:", df['sentiment'].nunique(), "unique")
