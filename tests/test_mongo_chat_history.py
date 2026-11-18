"""Automated tests verifying Chat History persistence in MongoDB."""

import os
import sys
import uuid
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.config.mongodb import get_database


def test_mongo_chat_history_workflow(client: TestClient):
    print("==========================================================")
    print("Starting MongoDB Chat History & Conversation Persistence Test")
    print("==========================================================")

    db = get_database()
    unique_id = uuid.uuid4().hex[:8]
    test_thread = f"test-thread-{unique_id}"

    # 1. Verify thread does not exist yet in MongoDB
    print(f"\n[1] Verifying thread '{test_thread}' is not initially in MongoDB...")
    # Clean up just in case
    client.delete(f"/chat/threads/{test_thread}")

    # 2. Send Turn 1
    print(f"\n[2] Sending Turn 1 to thread '{test_thread}'...")
    turn1_payload = {
        "text": "Hello, I want to check my account",
        "thread_id": test_thread,
        "channel": "rest",
        "context": {"username": "Alex"},
    }
    res1 = client.post("/chat", json=turn1_payload)
    assert res1.status_code == 200, f"Turn 1 failed: {res1.text}"
    data1 = res1.json()
    print("Turn 1 Bot Response:", [m["text"] for m in data1["bot_message"]])
    assert data1["thread_id"] == test_thread
    print("✓ Turn 1 HTTP 200 OK.")

    # 3. Check MongoDB directly for document persistence
    print("\n[3] Checking MongoDB directly for saved thread document...")
    # Call GET /chat/state to verify through API
    state_res = client.get(f"/chat/state/{test_thread}")
    assert state_res.status_code == 200
    state_data = state_res.json()
    assert state_data["thread_id"] == test_thread
    assert len(state_data.get("history", [])) == 1
    assert state_data["history"][0]["user"] == "Hello, I want to check my account"
    print("✓ Thread document verified with 1 saved turn in MongoDB.")

    # 4. Send Turn 2
    print(f"\n[4] Sending Turn 2 to thread '{test_thread}'...")
    turn2_payload = {
        "text": "My account ID is ACC-5544",
        "thread_id": test_thread,
        "channel": "rest",
        "context": {},
    }
    res2 = client.post("/chat", json=turn2_payload)
    assert res2.status_code == 200
    data2 = res2.json()
    print("Turn 2 Bot Response:", [m["text"] for m in data2["bot_message"]])
    print("✓ Turn 2 HTTP 200 OK.")

    # 5. Verify 2 turns in history
    print("\n[5] Verifying 2 turns in history...")
    state_res2 = client.get(f"/chat/state/{test_thread}")
    assert state_res2.status_code == 200
    state_data2 = state_res2.json()
    assert len(state_data2.get("history", [])) == 2
    print("✓ Turn 2 successfully appended to history array in MongoDB.")

    # 6. List active threads from MongoDB
    print("\n[6] Listing threads via GET /chat/threads...")
    threads_res = client.get("/chat/threads")
    assert threads_res.status_code == 200
    thread_list = threads_res.json()
    found = [t for t in thread_list if t["thread_id"] == test_thread]
    assert len(found) == 1
    print(f"Found Thread Summary: {found[0]}")
    assert found[0]["turns_count"] == 2
    print("✓ Active thread properly listed with turn count.")

    # 7. Delete individual message turn at index 0
    print("\n[7] Deleting message at index 0...")
    del_msg_res = client.delete(f"/chat/{test_thread}/message/0")
    assert del_msg_res.status_code == 200
    assert del_msg_res.json()["status"] == "deleted"

    state_after_del = client.get(f"/chat/state/{test_thread}").json()
    assert len(state_after_del["history"]) == 1
    assert state_after_del["history"][0]["user"] == "My account ID is ACC-5544"
    print("✓ Message 0 successfully deleted from MongoDB document.")

    # 8. Delete full thread
    print(f"\n[8] Deleting entire thread '{test_thread}'...")
    del_thread_res = client.delete(f"/chat/threads/{test_thread}")
    assert del_thread_res.status_code == 200
    assert del_thread_res.json()["status"] == "deleted"

    # Verify thread is removed from listing
    threads_after = client.get("/chat/threads").json()
    assert not any(t["thread_id"] == test_thread for t in threads_after)
    print("✓ Thread successfully deleted from MongoDB.")

    print("\n==========================================================")
    print("ALL MONGODB CHAT HISTORY TESTS PASSED SUCCESSFULLY!")
    print("==========================================================")


if __name__ == "__main__":
    with TestClient(app) as test_client:
        test_mongo_chat_history_workflow(test_client)
