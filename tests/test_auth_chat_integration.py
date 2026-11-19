"""Integration test suite verifying user authentication, JWT-isolated conversations, and token lifecycle in MongoDB."""

import sys
import uuid
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app
from app.services.llm_service import LLMService


def test_auth_chat_user_isolation_lifecycle(client: TestClient):
    """Verify end-to-end user registration, authenticated chat, user-isolated threads, and logout."""
    suffix = uuid.uuid4().hex[:6]
    user_a_email = f"user_a_{suffix}@example.com"
    user_b_email = f"user_b_{suffix}@example.com"
    password = "SecurePassword123!"

    # 1. Register User A and User B
    reg_a = client.post("/auth/register", json={"email": user_a_email, "password": password, "full_name": "Alice Developer"})
    assert reg_a.status_code == 201, f"User A registration failed: {reg_a.text}"
    user_a_id = reg_a.json()["id"]

    reg_b = client.post("/auth/register", json={"email": user_b_email, "password": password, "full_name": "Bob Tester"})
    assert reg_b.status_code == 201, f"User B registration failed: {reg_b.text}"
    user_b_id = reg_b.json()["id"]

    # 2. Login User A to get JWT tokens
    login_a = client.post("/auth/login", json={"email": user_a_email, "password": password})
    assert login_a.status_code == 200
    token_a = login_a.json()["access_token"]
    refresh_a = login_a.json()["refresh_token"]

    # Login User B to get JWT tokens
    login_b = client.post("/auth/login", json={"email": user_b_email, "password": password})
    assert login_b.status_code == 200
    token_b = login_b.json()["access_token"]

    # 3. Verify /auth/me for User A
    me_resp = client.get("/auth/me", headers={"Authorization": f"Bearer {token_a}"})
    assert me_resp.status_code == 200
    assert me_resp.json()["email"] == user_a_email
    assert me_resp.json()["id"] == user_a_id

    # 4. User A sends chat message with Bearer token (mocking Gemini call for instant local execution)
    thread_a = f"alice-thread-{suffix}"
    mock_turn = ([{"text": "Hello, this is a test response."}], {}, "general_chat", 0.95, {})

    with patch.object(LLMService, "_call_gemini", new_callable=AsyncMock, return_value=mock_turn):
        chat_resp_a = client.post(
            "/chat",
            headers={"Authorization": f"Bearer {token_a}"},
            json={
                "text": "Hello, this is Alice's private financial consultation",
                "thread_id": thread_a,
                "channel": "rest",
                "context": {"username": "Alice Developer"},
            },
        )
        assert chat_resp_a.status_code == 200

        # 5. User B sends chat message with Bearer token
        thread_b = f"bob-thread-{suffix}"
        chat_resp_b = client.post(
            "/chat",
            headers={"Authorization": f"Bearer {token_b}"},
            json={
                "text": "Hi, this is Bob's private inventory question",
                "thread_id": thread_b,
                "channel": "rest",
                "context": {"username": "Bob Tester"},
            },
        )
        assert chat_resp_b.status_code == 200

    # 6. Verify User Isolation in /chat/threads
    # User A listing threads should see thread_a but NOT thread_b
    threads_a = client.get("/chat/threads", headers={"Authorization": f"Bearer {token_a}"}).json()
    assert any(t["thread_id"] == thread_a for t in threads_a), "User A should see their own thread"
    assert not any(t["thread_id"] == thread_b for t in threads_a), "User A should NOT see User B's thread"

    # User B listing threads should see thread_b but NOT thread_a
    threads_b = client.get("/chat/threads", headers={"Authorization": f"Bearer {token_b}"}).json()
    assert any(t["thread_id"] == thread_b for t in threads_b), "User B should see their own thread"
    assert not any(t["thread_id"] == thread_a for t in threads_b), "User B should NOT see User A's thread"
    print("✓ User-isolated thread queries verified successfully.")

    # 7. Verify Token Refresh
    ref_resp = client.post("/auth/refresh", json={"refresh_token": refresh_a})
    assert ref_resp.status_code == 200
    new_token_a = ref_resp.json()["access_token"]
    new_refresh_a = ref_resp.json()["refresh_token"]

    # New token should be valid
    me_new = client.get("/auth/me", headers={"Authorization": f"Bearer {new_token_a}"})
    assert me_new.status_code == 200
    print("✓ Token refresh and rotation verified.")

    # 8. User A Logout
    logout_resp = client.post("/auth/logout", json={"refresh_token": new_refresh_a})
    assert logout_resp.status_code == 200

    # Trying to refresh with revoked token should fail
    ref_revoked = client.post("/auth/refresh", json={"refresh_token": new_refresh_a})
    assert ref_revoked.status_code == 401
    print("✓ Logout and refresh token revocation verified.")

    # 9. Clean up conversation threads
    client.delete(f"/chat/threads/{thread_a}")
    client.delete(f"/chat/threads/{thread_b}")
    print("✓ Cleaned up test threads.")


if __name__ == "__main__":
    with TestClient(app) as test_client:
        test_auth_chat_user_isolation_lifecycle(test_client)
    print("All auth-chat integration tests passed successfully!")
