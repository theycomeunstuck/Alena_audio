import sys
import requests


BASE_URL = "http://127.0.0.1:8099"
SECRET_PHRASE = "фиолетовый бегемот ест варенье на Луне"


def fail(message: str) -> None:
    print(f"[FAIL] {message}")
    sys.exit(1)


def ok(message: str) -> None:
    print(f"[OK]   {message}")


def main() -> None:
    print("=" * 60)
    print("TeachCopilot HTTP Server Test")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
    except requests.RequestException as exc:
        fail(f"/health request failed: {exc}")

    if response.status_code != 200:
        fail(f"/health returned HTTP {response.status_code}")

    data = response.json()
    if data.get("status") != "ok":
        fail(f"/health returned unexpected body: {data}")

    ok("/health works")

    response = requests.get(f"{BASE_URL}/v1/models", timeout=10)
    if response.status_code != 200:
        fail(f"/v1/models returned HTTP {response.status_code}")

    models = response.json().get("data", [])
    model_ids = [model.get("id") for model in models]

    if "teachcopilot-rag" not in model_ids:
        fail(f"teachcopilot-rag not found in /v1/models: {model_ids}")

    ok("/v1/models contains teachcopilot-rag")

    response = requests.post(
        f"{BASE_URL}/rag/search",
        json={
            "query": "какое секретное слово TeachCopilot",
            "limit": 3,
        },
        timeout=60,
    )

    if response.status_code != 200:
        fail(f"/rag/search returned HTTP {response.status_code}: {response.text}")

    rag_data = response.json()
    rag_text = str(rag_data)

    if SECRET_PHRASE not in rag_text:
        fail("/rag/search did not return the secret phrase")

    ok("/rag/search returns secret RAG phrase")

    response = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": "teachcopilot-rag",
            "messages": [
                {
                    "role": "user",
                    "content": "какое секретное слово TeachCopilot?",
                }
            ],
            "temperature": 0.1,
        },
        timeout=180,
    )

    if response.status_code != 200:
        fail(
            "/v1/chat/completions returned "
            f"HTTP {response.status_code}: {response.text}"
        )

    chat_data = response.json()
    answer = (
        chat_data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )

    if SECRET_PHRASE not in answer:
        fail(
            "/v1/chat/completions did not include the secret phrase. "
            f"Answer was: {answer}"
        )

    ok("/v1/chat/completions uses RAG context end-to-end")

    print("=" * 60)
    print("ALL SERVER TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()