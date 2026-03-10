from intent_classifier import IntentClassifier

classifier = IntentClassifier()

print("Intent Detection Test")
print("---------------------")

while True:

    query = input("Enter query: ")

    if query.lower() == "exit":
        break

    intent = classifier.detect_intent(query)

    print(f"Detected Intent: {intent}")