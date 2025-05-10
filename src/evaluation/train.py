import os
import requests
import base64
import time
import datetime

LOG_FILE = "registration_times.csv"


def submit_gei_to_api(gei_dir, api_url):
    gei_files = [f for f in os.listdir(gei_dir)]

    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, "w") as f:
            f.write("timestamp,file,subject_id,response_time,status\n")

    for gei_file in gei_files:
        gei_path = os.path.join(gei_dir, gei_file)
        subject_id = gei_file[:3]

        with open(gei_path, "rb") as img_file:
            encoded_image = base64.b64encode(img_file.read()).decode("utf-8")

        data = {"gei": encoded_image, "label": subject_id}

        start_time = time.time()
        response = requests.post(api_url, json=data)
        end_time = time.time()
        response_time = end_time - start_time

        timestamp = datetime.datetime.now().isoformat()

        if response.status_code == 200:
            status = "success"
        else:
            status = f"error_{response.status_code}"

        with open(LOG_FILE, "a") as log_file:
            log_file.write(f"{timestamp},{gei_file},{subject_id},{response_time},{status}\n")


if __name__ == "__main__":
    TRAINING_GEI_DIR = "output/train"
    API_URL = "http://127.0.0.1:5001/api/register"

    submit_gei_to_api(TRAINING_GEI_DIR, API_URL, LOG_FILE)
