"""FastAPI route definitions for Chat and NLU endpoints with MongoDB persistence."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, status
from fastapi.security import HTTPAuthorizationCredentials
from app.controllers.chat_controller import ChatController
from app.schemas.chat_schema import ChatRequest, ChatResponse
from app.services.auth_service import security_bearer, decode_token

router = APIRouter(tags=["Chat"])

# Singleton controller dependency
_controller_instance: ChatController | None = None


def get_chat_controller() -> ChatController:
    """Dependency provider for ChatController instance."""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = ChatController()
    return _controller_instance


def get_optional_user_id(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security_bearer),
) -> Optional[str]:
    """Extract user_id from optional Bearer token if provided."""
    if credentials and credentials.credentials:
        try:
            payload = decode_token(credentials.credentials, expected_type="access")
            return payload.get("sub")
        except Exception:
            return None
    return None


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Process user message and return conversational state",
    description="Primary chat endpoint accepting user message and returning NLU intent, entities, and bot messages. Automatically persists to MongoDB.",
)
async def post_chat_message(
    request: ChatRequest,
    controller: ChatController = Depends(get_chat_controller),
    user_id: Optional[str] = Depends(get_optional_user_id),
) -> ChatResponse:
    """Send a user message to the chatbot and receive bot messages and state."""
    return await controller.handle_message(request, user_id=user_id)


@router.post(
    "/chat/stream",
    status_code=status.HTTP_200_OK,
    summary="Process user message and stream response token-by-token (SSE)",
    description="Streams Gemini conversational tokens in real time via Server-Sent Events (SSE), then persists completed turn to MongoDB.",
)
def post_chat_message_stream(
    request: ChatRequest,
    controller: ChatController = Depends(get_chat_controller),
    user_id: Optional[str] = Depends(get_optional_user_id),
):
    """Send a user message and receive real-time streamed tokens followed by final state."""
    return controller.handle_message_stream(request, user_id=user_id)


@router.get(
    "/chat/state/{thread_id}",
    summary="Retrieve current session state for a thread",
    description="Returns current context and conversation history from MongoDB for the specified thread_id.",
)
async def get_chat_state(
    thread_id: str,
    controller: ChatController = Depends(get_chat_controller),
    user_id: Optional[str] = Depends(get_optional_user_id),
) -> Dict[str, Any]:
    """Get the current conversation memory state for a thread."""
    return await controller.get_thread_state(thread_id, user_id=user_id)


@router.get(
    "/chat/threads",
    summary="List all active chat conversations",
    description="Returns an array of all active chat sessions stored in MongoDB with titles and message counts.",
)
async def list_chats(
    controller: ChatController = Depends(get_chat_controller),
    user_id: Optional[str] = Depends(get_optional_user_id),
) -> List[Dict[str, Any]]:
    """List all active chat conversations."""
    return await controller.list_threads(user_id=user_id)


@router.delete(
    "/chat/threads/{thread_id}",
    summary="Delete a particular chat conversation",
    description="Permanently deletes a specific chat thread and its full conversation history from MongoDB.",
)
async def delete_particular_chat(
    thread_id: str,
    controller: ChatController = Depends(get_chat_controller),
) -> Dict[str, Any]:
    """Delete a particular chat conversation."""
    return await controller.delete_thread(thread_id)


@router.delete(
    "/chat/{thread_id}/message/{message_index}",
    summary="Delete a particular message from a chat",
    description="Deletes an individual message/turn by index from a specific chat conversation in MongoDB.",
)
async def delete_particular_message(
    thread_id: str,
    message_index: int,
    controller: ChatController = Depends(get_chat_controller),
) -> Dict[str, Any]:
    """Delete a particular message from a chat."""
    return await controller.delete_message(thread_id, message_index)


@router.delete(
    "/chat/state/{thread_id}",
    summary="Reset or delete a particular chat thread",
    description="Clears conversation memory and resets awaiting state for the thread_id in MongoDB.",
)
async def delete_chat_state(
    thread_id: str,
    controller: ChatController = Depends(get_chat_controller),
) -> Dict[str, Any]:
    """Reset or delete thread state."""
    return await controller.delete_thread(thread_id)
