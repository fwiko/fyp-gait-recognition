import os
import requests
import base64
import json
import time

def test_gei_classification(test_dir, api_url):
    gei_files = [f for f in os.listdir(test_dir)]
    results = []

    for gei_file in gei_files:
        gei_path = os.path.join(test_dir, gei_file)

        true_label = gei_file[:3]

        with open(gei_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

        data = {"gei": encoded_image}

        start_time = time.time()
        response = requests.post(api_url, json=data)
        end_time = time.time()
        response_time = end_time - start_time

        if response.status_code == 200:
            result = response.json()
            predicted_label = result.get("person")
            confidence = result.get("confidence", 0)

            results.append(
                {
                    "file": gei_file,
                    "true_label": true_label,
                    "predicted_label": predicted_label,
                    "confidence": confidence,
                    "correct": predicted_label == true_label,
                    "above_threshold": confidence >= 75.0,
                    "response_time": response_time,
                }
            )
        else:
            results.append(
                {
                    "file": gei_file,
                    "true_label": true_label,
                    "predicted_label": None,
                    "confidence": 0,
                    "correct": False,
                    "above_threshold": False,
                    "response_time": response_time,
                    "error": f"Error: {response.status_code}",
                }
            )

    with open("classification_results.json", "w") as out:
        json.dump(results, out)


if __name__ == "__main__":
    TEST_GEI_DIR = "output/test"
    API_URL = "http://127.0.0.1:5001/api/classify"
    OUTPUT_FILE = "classification_results.json"

    test_gei_classification(TEST_GEI_DIR, API_URL, OUTPUT_FILE)
