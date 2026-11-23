import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    """Verify health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "ai-chat-api"
    assert data["provider"] == "gemini"


def test_root_endpoint():
    """Verify root endpoint serves the UI HTML or API info."""
    response = client.get("/")
    assert response.status_code == 200
    assert "Admin Panel" in response.text or "AI Chat API" in response.text


def test_ui_endpoint():
    """Verify /ui endpoint provides information pointing to the frontend."""
    response = client.get("/ui")
    assert response.status_code == 200
    data = response.json()
    assert "frontend" in data["message"].lower() or "frontend_url" in data


def test_multi_turn_order_status_conversation():
    """Verify the 2-turn conversation flow matching the UI Admin Panel screenshot."""
    thread_id = "test-user"

    # Reset thread before test
    client.delete(f"/chat/state/{thread_id}")

    # Turn 1: Ask for order status
    turn1_response = client.post(
        "/chat",
        json={
            "text": "can you help to find the status of my order",
            "thread_id": thread_id,
            "channel": "rest",
            "context": {"username": "Admin"},
        },
    )
    assert turn1_response.status_code == 200
    turn1_data = turn1_response.json()

    assert turn1_data["thread_id"] == thread_id
    assert turn1_data["nlu"]["intent"]["intent"] == "check_order_status"
    turn1_bot_texts = " ".join([m["text"] for m in turn1_data["bot_message"]]).lower()
    assert any(w in turn1_bot_texts for w in ["sure", "help", "order", "number", "certainly"])

    # Turn 2: Provide order number ORD987124
    turn2_response = client.post(
        "/chat",
        json={
            "text": "it's ORD987124",
            "thread_id": thread_id,
            "channel": "rest",
            "context": {},
        },
    )
    assert turn2_response.status_code == 200
    turn2_data = turn2_response.json()

    assert turn2_data["thread_id"] == thread_id
    assert turn2_data["user_message"]["text"] == "it's ORD987124"
    assert turn2_data["nlu"]["intent"]["intent"] == "check_order_status"
    assert turn2_data["nlu"]["intent"]["confidence"] > 0.4

    # Check bot messages
    turn2_bot_texts = " ".join([m["text"] for m in turn2_data["bot_message"]])
    assert "ORD987124" in turn2_bot_texts or "ORD987124" in str(turn2_data["nlu"]) or "shipped" in turn2_bot_texts.lower()


def test_greeting_intent():
    """Verify greeting intent detection."""
    response = client.post(
        "/chat",
        json={"text": "Hello there!", "thread_id": "greet-test"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["nlu"]["intent"]["intent"] == "greeting"
    assert len(data["bot_message"]) >= 1


def test_delete_particular_chat():
    """Verify deleting a particular chat conversation thread."""
    thread_id = "test-delete-thread"

    # Send a message to create the thread
    client.post("/chat", json={"text": "Creating test thread", "thread_id": thread_id})
    
    # Verify thread exists in list
    threads_resp = client.get("/chat/threads")
    assert threads_resp.status_code == 200
    threads = threads_resp.json()
    assert any(t["thread_id"] == thread_id for t in threads)

    # Delete this particular chat
    del_resp = client.delete(f"/chat/threads/{thread_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # Verify thread is no longer in list
    threads_after = client.get("/chat/threads").json()
    assert not any(t["thread_id"] == thread_id for t in threads_after)


def test_delete_particular_message():
    """Verify deleting a specific message from a conversation thread."""
    thread_id = "test-msg-del-thread"

    # Send two messages
    client.post("/chat", json={"text": "First message", "thread_id": thread_id})
    client.post("/chat", json={"text": "Second message", "thread_id": thread_id})

    # Check state has 2 turns
    state_resp = client.get(f"/chat/state/{thread_id}")
    assert len(state_resp.json()["history"]) == 2

    # Delete the first message (index 0)
    del_msg_resp = client.delete(f"/chat/{thread_id}/message/0")
    assert del_msg_resp.status_code == 200
    assert del_msg_resp.json()["status"] == "deleted"

    # Check state now has 1 turn
    state_after = client.get(f"/chat/state/{thread_id}")
    assert len(state_after.json()["history"]) == 1
    assert state_after.json()["history"][0]["user"] == "Second message"


def test_chromadb_knowledge_base():
    """Verify ChromaDB document management and vector query endpoints."""
    # 1. List seeded knowledge
    res_list = client.get("/knowledge")
    assert res_list.status_code == 200
    data = res_list.json()
    assert data["status"] == "success"
    assert data["total_documents"] >= 4

    # 2. Add custom document
    doc_payload = {
        "text": "Premium VIP members receive free express overnight delivery on all electronics.",
        "title": "VIP Delivery Policy",
        "category": "membership",
        "source": "vip_perks.pdf",
        "id": "vip_perk_doc_01"
    }
    res_add = client.post("/knowledge", json=doc_payload)
    assert res_add.status_code == 201
    assert res_add.json()["status"] == "success"
    assert res_add.json()["doc_id"] == "vip_perk_doc_01"

    # 3. Semantic vector search query
    query_payload = {
        "query": "How fast do VIP members get electronic shipments?",
        "n_results": 2
    }
    res_query = client.post("/knowledge/query", json=query_payload)
    assert res_query.status_code == 200
    query_data = res_query.json()
    assert query_data["results_count"] >= 1
    # Check top hit text mentions VIP or delivery
    top_hit = query_data["results"][0]
    assert "VIP" in top_hit["text"] or "delivery" in top_hit["text"]

    # 4. Clean up / Delete document
    res_del = client.delete("/knowledge/vip_perk_doc_01")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "success"


def test_rag_retrieval_in_chat():
    """Verify that asking a knowledge base question retrieves context in chat response."""
    chat_payload = {
        "text": "What is your return policy and refund window?",
        "thread_id": "rag-test-user",
        "channel": "rest",
    }
    res = client.post("/chat", json=chat_payload)
    assert res.status_code == 200
    data = res.json()
    # Check that ChromaDB retrieved context is included in state context
    retrieved = data["context"].get("retrieved_context", [])
    assert len(retrieved) > 0
    assert any("return" in r.get("text", "").lower() for r in retrieved)


if __name__ == "__main__":
    test_health_check()
    test_root_endpoint()
    test_ui_endpoint()
    test_multi_turn_order_status_conversation()
    test_greeting_intent()
    test_delete_particular_chat()
    test_delete_particular_message()
    test_chromadb_knowledge_base()
    test_rag_retrieval_in_chat()
    print("All tests passed successfully!")
