import requests
import base64
import json
from typing import Dict, Optional, Tuple


class ServerAPIClient:
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url.rstrip("/")
        self.register_endpoint = f"{self.base_url}/api/register"
        self.classify_endpoint = f"{self.base_url}/api/classify"

    def register_gei(self, gei_image: bytes, label: str) -> Tuple[bool, Optional[str]]:
        try:
            # Encode the image to base64
            gei_base64 = base64.b64encode(gei_image).decode("utf-8")

            # Prepare the payload
            payload = {"label": label, "gei": gei_base64}

            # Make the request
            response = requests.post(self.register_endpoint, json=payload)

            # Check if the request was successful
            if response.status_code == 200:
                return True, None
            else:
                return (
                    False,
                    f"Server returned status code {response.status_code}: {response.text}",
                )

        except requests.exceptions.RequestException as e:
            return False, f"Request failed: {str(e)}"
        except Exception as e:
            return False, f"Unexpected error: {str(e)}"

    def classify_gei(self, gei_image: bytes) -> Tuple[Optional[Dict], Optional[str]]:
        try:
            # Encode the image to base64
            gei_base64 = base64.b64encode(gei_image).decode("utf-8")

            # Prepare the payload
            payload = {"gei": gei_base64}

            # Make the request
            response = requests.post(self.classify_endpoint, json=payload)

            # Check if the request was successful
            if response.status_code == 200:
                return response.json(), None
            else:
                return (
                    None,
                    f"Server returned status code {response.status_code}: {response.text}",
                )

        except requests.exceptions.RequestException as e:
            return None, f"Request failed: {str(e)}"
        except json.JSONDecodeError:
            return None, "Failed to parse server response"
        except Exception as e:
            return None, f"Unexpected error: {str(e)}"
