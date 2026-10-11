import os
import time

import requests

BASE_URL = os.getenv("API_URL", "http://localhost:8000")


def run_smoke_test():
    print(f"Starting smoke test against {BASE_URL}...")

    # 1. /health
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200, f"Healthcheck failed: {resp.text}"

    # 2. /register
    user_payload = {"user_id": "smoke_tester", "password": "securepassword"}
    resp = requests.post(f"{BASE_URL}/register", json=user_payload)
    assert resp.status_code in (201, 400), f"Registration failed: {resp.text}"

    # 3. /login
    login_data = {"username": "smoke_tester", "password": "securepassword"}
    resp = requests.post(f"{BASE_URL}/login", data=login_data)
    assert resp.status_code == 200, f"Login failed: {resp.text}"

    token = resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 4. /documents
    docs_payload = [
        {
            "text": "Smoke test document ensuring end-to-end functionality.",
            "metadata": {"source": "smoke"},
        }
    ]
    resp = requests.post(f"{BASE_URL}/documents", json=docs_payload, headers=headers)
    assert resp.status_code == 200, f"Document insertion failed: {resp.text}"

    # 5. /query
    query_payload = {"question": "What is the smoke test document?", "k": 1}
    resp = requests.post(f"{BASE_URL}/query", json=query_payload, headers=headers)
    assert resp.status_code == 200, f"Query failed: {resp.text}"

    print("Smoke test passed: All critical endpoints are functional.")


if __name__ == "__main__":
    # Wait for the service to boot before testing
    for _ in range(15):
        try:
            if requests.get(f"{BASE_URL}/health").status_code == 200:
                break
        except requests.ConnectionError:
            time.sleep(2)
    run_smoke_test()
