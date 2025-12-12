import json
import os
from datetime import datetime

def save_text_to_json(text):
    if not os.path.exists("outputs"):
        os.makedirs("outputs")

    filename = f"outputs/extracted_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    data = {"extracted_text": text}

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    return filename
