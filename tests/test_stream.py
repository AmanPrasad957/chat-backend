"""Automated test suite for Server-Sent Events (SSE) /chat/stream endpoint."""

import sys
import json
import uuid
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_chat_stream_sse_flow():
    """Verify that /chat/stream yields SSE events (start, token, done) and persists to MongoDB."""
    thread_id = f"test-stream-{uuid.uuid4().hex[:8]}"

    # Send streaming chat request
    payload = {
        "text": "Hello, how does shipping work?",
        "thread_id": thread_id,
        "channel": "rest",
        "context": {"username": "StreamUser"},
    }

    response = client.post("/chat/stream", json=payload)
    assert response.status_code == 200
    assert "text/event-stream" in response.headers.get("content-type", "")

    # Parse SSE stream chunks
    raw_lines = response.text.split("\n\n")
    events = []
    accumulated_tokens = []

    for block in raw_lines:
        line = block.strip()
        if not line.startswith("data:"):
            continue
        json_str = line[5:].strip()
        if not json_str:
            continue
        try:
            ev = json.loads(json_str)
            events.append(ev)
            if ev.get("type") == "token":
                accumulated_tokens.append(ev.get("delta", ""))
        except Exception as e:
            print("Failed to parse SSE JSON chunk:", line, e)

    # 1. Verify event types
    event_types = [e.get("type") for e in events]
    assert "start" in event_types, f"Expected 'start' in events: {event_types}"
    assert "token" in event_types, f"Expected 'token' in events: {event_types}"
    assert "done" in event_types, f"Expected 'done' in events: {event_types}"

    # 2. Verify tokens were streamed
    full_streamed_text = "".join(accumulated_tokens)
    assert len(full_streamed_text.strip()) > 0, "Expected non-empty streamed text"
    print(f"✓ Streamed {len(accumulated_tokens)} tokens: '{full_streamed_text[:60]}...'")

    # 3. Verify done event payload
    done_event = next(e for e in events if e.get("type") == "done")
    assert done_event["thread_id"] == thread_id
    assert "bot_message" in done_event
    assert "nlu" in done_event
    assert "context" in done_event

    # 4. Verify MongoDB state persistence
    state_resp = client.get(f"/chat/state/{thread_id}")
    assert state_resp.status_code == 200
    state_data = state_resp.json()
    assert state_data["thread_id"] == thread_id
    assert len(state_data["history"]) >= 1
    assert state_data["history"][0]["user"] == "Hello, how does shipping work?"
    print("✓ MongoDB successfully persisted streamed turn.")

    # 5. Clean up thread
    del_resp = client.delete(f"/chat/threads/{thread_id}")
    assert del_resp.status_code == 200
    print(f"✓ Cleaned up thread '{thread_id}'.")


def test_chat_stream_order_tracking():
    """Verify streaming response for order number tracking."""
    thread_id = f"test-order-stream-{uuid.uuid4().hex[:8]}"

    payload = {
        "text": "Check status of order ORD987124",
        "thread_id": thread_id,
        "channel": "rest",
        "context": {"username": "Admin"},
    }

    response = client.post("/chat/stream", json=payload)
    assert response.status_code == 200

    raw_lines = response.text.split("\n\n")
    tokens = []
    done_event = None

    for block in raw_lines:
        line = block.strip()
        if line.startswith("data:"):
            data = json.loads(line[5:].strip())
            if data.get("type") == "token":
                tokens.append(data.get("delta", ""))
            elif data.get("type") == "done":
                done_event = data

    full_text = "".join(tokens)
    assert done_event is not None
    assert "ORD987124" in full_text or "ORD987124" in str(done_event.get("nlu", {}))
    print(f"✓ Order status stream output: '{full_text[:60]}...'")

    # Clean up
    client.delete(f"/chat/threads/{thread_id}")


if __name__ == "__main__":
    test_chat_stream_sse_flow()
    test_chat_stream_order_tracking()
    print("All SSE streaming tests passed successfully!")
