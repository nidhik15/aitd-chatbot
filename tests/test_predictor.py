from feedback_predictor import CollegeFeedbackPredictor

predictor = CollegeFeedbackPredictor()
tests = [
    "website needs improvement",  # → Valid, Website, Negative
    "Nothing",                    # → Invalid, Uncategorized, Neutral
    "canteen is good"             # → Valid, Canteen, Positive
]

for test in tests:
    print(predictor.predict(test))
