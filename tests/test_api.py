from fastapi.testclient import TestClient

from api.app import app, engine


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_memory_endpoint_accepts_message():
    response = client.post(
        "/memory",
        json={"message": "I use Arch Linux."},
    )

    assert response.status_code == 200
    assert response.json()["status"] in {
        "stored",
        "updated",
        "ignored",
    }


def test_recall_endpoint_returns_memories_key():
    response = client.post(
        "/recall",
        json={
            "query": "What operating system does the user use?",
            "top_k": 5,
        },
    )

    assert response.status_code == 200
    assert "memories" in response.json()
    assert isinstance(response.json()["memories"], list)


def test_list_memories_endpoint():
    response = client.get("/memories")

    assert response.status_code == 200
    assert "memories" in response.json()


def test_unknown_memory_can_return_empty_result():
    response = client.post(
        "/recall",
        json={
            "query": "What car does the user drive?",
            "top_k": 5,
            "threshold": 0.45,
        },
    )

    assert response.status_code == 200
    assert response.json()["memories"] == []
