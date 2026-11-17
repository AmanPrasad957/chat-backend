"""FastAPI route definitions for Chat and NLU endpoints."""

from typing import Any, Dict, List
from fastapi import APIRouter, Depends, status
from app.controllers.chat_controller import ChatController
from app.schemas.chat_schema import ChatRequest, ChatResponse

router = APIRouter(tags=["Chat"])

# Singleton controller dependency
_controller_instance: ChatController | None = None


def get_chat_controller() -> ChatController:
    """Dependency provider for ChatController instance."""
    global _controller_instance
    if _controller_instance is None:
        _controller_instance = ChatController()
    return _controller_instance


@router.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Process user message and return conversational state",
    description="Primary chat endpoint accepting user message and returning NLU intent, entities, and bot messages.",
)
async def post_chat_message(
    request: ChatRequest,
    controller: ChatController = Depends(get_chat_controller),
) -> ChatResponse:
    """Send a user message to the chatbot and receive bot messages and state."""
    return await controller.handle_message(request)


@router.get(
    "/chat/state/{thread_id}",
    summary="Retrieve current session state for a thread",
    description="Returns current context and conversation history for the specified thread_id.",
)
async def get_chat_state(
    thread_id: str,
    controller: ChatController = Depends(get_chat_controller),
) -> Dict[str, Any]:
    """Get the current conversation memory state for a thread."""
    return controller.get_thread_state(thread_id)


@router.get(
    "/chat/threads",
    summary="List all active chat conversations",
    description="Returns an array of all active chat sessions with titles and message counts.",
)
async def list_chats(
    controller: ChatController = Depends(get_chat_controller),
) -> List[Dict[str, Any]]:
    """List all active chat conversations."""
    return controller.list_threads()


@router.delete(
    "/chat/threads/{thread_id}",
    summary="Delete a particular chat conversation",
    description="Deletes a specific chat thread and its full conversation history.",
)
async def delete_particular_chat(
    thread_id: str,
    controller: ChatController = Depends(get_chat_controller),
) -> Dict[str, Any]:
    """Delete a particular chat conversation."""
    return controller.delete_thread(thread_id)


@router.delete(
    "/chat/{thread_id}/message/{message_index}",
    summary="Delete a particular message from a chat",
    description="Deletes an individual message/turn by index from a specific chat conversation.",
)
async def delete_particular_message(
    thread_id: str,
    message_index: int,
    controller: ChatController = Depends(get_chat_controller),
) -> Dict[str, Any]:
    """Delete a particular message from a chat."""
    return controller.delete_message(thread_id, message_index)


@router.delete(
    "/chat/state/{thread_id}",
    summary="Reset or delete a particular chat thread",
    description="Clears conversation memory and resets awaiting state for the thread_id.",
)
async def delete_chat_state(
    thread_id: str,
    controller: ChatController = Depends(get_chat_controller),
) -> Dict[str, Any]:
    """Reset or delete thread state."""
    return controller.delete_thread(thread_id)
