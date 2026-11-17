"""Automated verification tests for MongoDB Authentication & Authorization (JWT Tokens)."""

import os
import uuid
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_auth_workflow(client: TestClient):
    print("==================================================")
    print("Starting MongoDB Authentication & JWT Verification")
    print("==================================================")

    # Use a unique email for every test run
    unique_suffix = uuid.uuid4().hex[:8]
    test_email = f"test_{unique_suffix}@example.com"
    test_password = "SecurePassword123!"
    test_name = "Alex Doe"

    # 1. Health check with MongoDB
    print("\n[1] Testing GET /health for MongoDB connection...")
    res = client.get("/health")
    assert res.status_code == 200, f"Health check failed: {res.text}"
    health_data = res.json()
    print(f"Health Response: {health_data}")
    assert health_data["mongodb_configured"] is True
    assert health_data["mongodb_connected"] is True, "MongoDB should be connected locally!"
    print("✓ MongoDB connection verified healthy.")

    # 2. Register new user
    print(f"\n[2] Testing POST /auth/register for {test_email}...")
    reg_payload = {
        "email": test_email,
        "password": test_password,
        "full_name": test_name,
    }
    res = client.post("/auth/register", json=reg_payload)
    assert res.status_code == 201, f"Registration failed ({res.status_code}): {res.text}"
    user_data = res.json()
    print(f"Registered User: {user_data}")
    assert user_data["email"] == test_email
    assert user_data["full_name"] == test_name
    assert "id" in user_data
    assert "hashed_password" not in user_data, "Password hash must not be exposed!"
    print("✓ User successfully registered with MongoDB.")

    # 3. Duplicate registration check
    print("\n[3] Testing duplicate email registration prevention...")
    dup_res = client.post("/auth/register", json=reg_payload)
    assert dup_res.status_code == 400, f"Expected 400 for duplicate, got {dup_res.status_code}"
    print("✓ Duplicate email correctly rejected with 400 Bad Request.")

    # 4. Invalid email format check
    print("\n[4] Testing invalid email format...")
    inv_res = client.post("/auth/register", json={"email": "not-an-email", "password": "password123"})
    assert inv_res.status_code == 422, f"Expected 422 for bad email, got {inv_res.status_code}"
    print("✓ Invalid email correctly rejected with 422.")

    # 5. Login with incorrect password
    print("\n[5] Testing login with wrong password...")
    bad_login_res = client.post(
        "/auth/login",
        json={"email": test_email, "password": "WrongPassword999!"},
    )
    assert bad_login_res.status_code == 401, f"Expected 401 for wrong password, got {bad_login_res.status_code}"
    print("✓ Wrong password rejected with 401 Unauthorized.")

    # 6. Login with correct password
    print("\n[6] Testing login with correct credentials...")
    login_res = client.post(
        "/auth/login",
        json={"email": test_email, "password": test_password},
    )
    assert login_res.status_code == 200, f"Login failed: {login_res.text}"
    token_data = login_res.json()
    print(f"Token Type: {token_data['token_type']} | Expires in: {token_data['expires_in']}s")
    assert "access_token" in token_data
    assert "refresh_token" in token_data
    assert token_data["token_type"] == "bearer"
    access_token = token_data["access_token"]
    refresh_token = token_data["refresh_token"]
    print("✓ Login succeeded. Received JWT access_token and refresh_token.")

    # 7. Access protected /auth/me with Bearer token
    print("\n[7] Testing GET /auth/me with Bearer access token...")
    me_res = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    assert me_res.status_code == 200, f"Failed to access /auth/me: {me_res.text}"
    me_data = me_res.json()
    print(f"Authenticated User Profile: {me_data}")
    assert me_data["email"] == test_email
    assert me_data["id"] == user_data["id"]
    print("✓ Protected /auth/me endpoint accessed successfully.")

    # 8. Access protected /auth/me without token
    print("\n[8] Testing GET /auth/me without token (unauthorized)...")
    unauth_res = client.get("/auth/me")
    assert unauth_res.status_code == 401, f"Expected 401, got {unauth_res.status_code}"
    print("✓ Missing token correctly rejected with 401 Unauthorized.")

    # 9. Token Refresh
    print("\n[9] Testing POST /auth/refresh with refresh token...")
    refresh_res = client.post(
        "/auth/refresh",
        json={"refresh_token": refresh_token},
    )
    assert refresh_res.status_code == 200, f"Refresh failed: {refresh_res.text}"
    new_token_data = refresh_res.json()
    new_access_token = new_token_data["access_token"]
    new_refresh_token = new_token_data["refresh_token"]
    assert new_access_token != access_token, "New access token should be generated"
    print("✓ Token refresh succeeded. Received fresh access token and rotated refresh token.")

    # 10. Verify new access token works on /auth/me
    print("\n[10] Testing GET /auth/me with new refreshed access token...")
    new_me_res = client.get(
        "/auth/me",
        headers={"Authorization": f"Bearer {new_access_token}"},
    )
    assert new_me_res.status_code == 200
    print("✓ Refreshed access token successfully authenticated.")

    # 11. Logout / Revoke token
    print("\n[11] Testing POST /auth/logout...")
    logout_res = client.post(
        "/auth/logout",
        json={"refresh_token": new_refresh_token},
    )
    assert logout_res.status_code == 200
    print(f"Logout Response: {logout_res.json()}")
    print("✓ Logout succeeded.")

    # 12. Try to refresh with the revoked token (should fail 401)
    print("\n[12] Testing POST /auth/refresh with revoked refresh token...")
    revoked_res = client.post(
        "/auth/refresh",
        json={"refresh_token": new_refresh_token},
    )
    assert revoked_res.status_code == 401, f"Expected 401 for revoked token, got {revoked_res.status_code}"
    print("✓ Revoked token correctly rejected with 401 Unauthorized.")

    # 13. Verify API v1 prefix works identically
    print("\n[13] Testing /api/v1/auth/login prefix...")
    v1_login = client.post(
        "/api/v1/auth/login",
        json={"email": test_email, "password": test_password},
    )
    assert v1_login.status_code == 200
    assert "access_token" in v1_login.json()
    print("✓ /api/v1/auth prefix verified.")

    print("\n==================================================")
    print("ALL MONGODB AUTH & JWT TESTS PASSED SUCCESSFULLY!")
    print("==================================================")


if __name__ == "__main__":
    with TestClient(app) as test_client:
        test_auth_workflow(test_client)
