import ollama
from intent_prompt import SYSTEM_PROMPT


class IntentClassifier:

    def __init__(self):
        self.model = "qwen2.5:0.5b"

    def detect_intent(self, user_query):

        response = ollama.chat(
            model=self.model,
            options={
                "temperature": 0,
                "top_p": 0.9,
                "num_predict": 20
            },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_query}
            ]
        )

        intent = response["message"]["content"].strip()

        if "Navigation" in intent:
            intent = "Navigation"
        elif "Feedback" in intent:
            intent = "Feedback"

        return intent