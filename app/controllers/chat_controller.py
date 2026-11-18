"""Chat controller orchestrating request parsing, LLM service interactions, and MongoDB persistence."""

import logging
from typing import Any, Dict, List, Optional
from app.schemas.chat_schema import (
    BotMessageItem,
    ChatRequest,
    ChatResponse,
    IntentModel,
    NLUModel,
    UserMessage,
)
from app.services.llm_service import LLMService

logger = logging.getLogger("ai_chat_api.chat_controller")


class ChatController:
    """Controller layer handling chat operations and bridging routes to services."""

    def __init__(self, llm_service: LLMService | None = None) -> None:
        self.llm_service = llm_service or LLMService()

    async def handle_message(self, request: ChatRequest, user_id: Optional[str] = None) -> ChatResponse:
        """Process an incoming chat message and return state response matching the admin interface."""
        text = request.text or ""
        thread_id = request.thread_id or "test-user"
        channel = request.channel or "rest"
        context = request.context or {}

        logger.info("Processing message for thread '%s' on channel '%s': %s", thread_id, channel, text)

        bot_messages, entities, intent, confidence, updated_context = (
            await self.llm_service.generate_chat_turn(
                thread_id=thread_id,
                user_text=text,
                channel=channel,
                context=context,
                user_id=user_id,
            )
        )

        user_message_obj = UserMessage(
            thread_id=thread_id,
            text=text,
            channel=channel,
            context=context,
        )

        bot_message_items = [
            BotMessageItem(text=msg.get("text", "") if isinstance(msg, dict) else str(msg))
            for msg in bot_messages
        ]

        nlu_obj = NLUModel(
            entities=entities,
            intent=IntentModel(intent=intent, confidence=confidence),
        )

        response = ChatResponse(
            thread_id=thread_id,
            user_message=user_message_obj,
            bot_message=bot_message_items,
            nlu=nlu_obj,
            context=updated_context,
        )

        return response

    async def get_thread_state(self, thread_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch current thread state and context from MongoDB."""
        return await self.llm_service.get_thread_state(thread_id, user_id)

    async def list_threads(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all active chat threads from MongoDB."""
        return await self.llm_service.list_all_threads(user_id)

    async def reset_thread(self, thread_id: str) -> Dict[str, Any]:
        """Reset conversation memory for a thread."""
        return await self.delete_thread(thread_id)

    async def delete_thread(self, thread_id: str) -> Dict[str, Any]:
        """Delete a particular conversation thread from MongoDB."""
        cleared = await self.llm_service.clear_thread_state(thread_id)
        return {
            "thread_id": thread_id,
            "status": "deleted" if cleared else "not_found",
            "message": f"Chat thread '{thread_id}' has been deleted.",
        }

    async def delete_message(self, thread_id: str, message_index: int) -> Dict[str, Any]:
        """Delete a particular message from a chat thread in MongoDB."""
        success = await self.llm_service.delete_message(thread_id, message_index)
        return {
            "thread_id": thread_id,
            "message_index": message_index,
            "status": "deleted" if success else "not_found",
            "message": f"Message {message_index} deleted from thread '{thread_id}'." if success else "Message not found.",
        }
