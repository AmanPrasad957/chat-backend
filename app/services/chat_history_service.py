"""Chat History persistence service with MongoDB storage and in-memory fallback."""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from app.config.mongodb import get_database

logger = logging.getLogger("ai_chat_api.chat_history_service")


class ChatHistoryService:
    """Async service managing conversation threads and message turns in MongoDB."""

    def __init__(self) -> None:
        # In-memory fallback for testing or when MongoDB is offline
        self._memory_fallback: Dict[str, Dict[str, Any]] = {}

    def _get_db_safe(self):
        """Attempt to retrieve the MongoDB database instance safely."""
        try:
            return get_database()
        except Exception:
            return None

    def _default_thread_dict(self, thread_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Construct a fresh default conversation thread dict."""
        now = datetime.now(timezone.utc)
        return {
            "thread_id": thread_id,
            "user_id": user_id or "anonymous",
            "title": "New Chat",
            "history": [],
            "awaiting": None,
            "context": {
                "result": {},
                "username": "Admin",
            },
            "created_at": now,
            "updated_at": now,
        }

    async def get_or_create_thread(self, thread_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Fetch conversation thread state from MongoDB; initialize if it doesn't exist."""
        db = self._get_db_safe()
        if db is not None:
            try:
                doc = await db.conversations.find_one({"thread_id": thread_id})
                if doc:
                    if "_id" in doc:
                        doc["_id"] = str(doc["_id"])
                    return doc

                # Create new thread document in MongoDB
                new_doc = self._default_thread_dict(thread_id, user_id)
                await db.conversations.insert_one(dict(new_doc))
                return new_doc
            except Exception as exc:
                logger.warning("MongoDB read failed for thread '%s': %s. Falling back to memory.", thread_id, exc)

        # Fallback to in-memory
        if thread_id not in self._memory_fallback:
            self._memory_fallback[thread_id] = self._default_thread_dict(thread_id, user_id)
        return self._memory_fallback[thread_id]

    async def save_turn(
        self,
        thread_id: str,
        user_text: str,
        bot_messages: List[Dict[str, str]],
        intent: str,
        entities: Dict[str, Any],
        context: Dict[str, Any],
        user_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Atomically append a completed dialogue turn and update conversation context in MongoDB."""
        now = datetime.now(timezone.utc)
        bot_texts = [
            m.get("text", "") if isinstance(m, dict) else str(m)
            for m in bot_messages
        ]

        turn = {
            "user": user_text,
            "bot": bot_texts,
            "intent": intent,
            "entities": entities,
            "timestamp": now,
        }

        # Calculate a descriptive title from first user message
        title_candidate = (user_text[:35] + "...") if len(user_text) > 35 else (user_text or "New Chat")

        db = self._get_db_safe()
        if db is not None:
            try:
                # Check current thread to decide whether to update title
                existing = await db.conversations.find_one({"thread_id": thread_id})
                update_fields: Dict[str, Any] = {
                    "updated_at": now,
                    "context": context,
                    "awaiting": None,
                }
                if user_id:
                    update_fields["user_id"] = user_id

                if not existing or existing.get("title") in ("New Chat", "New Conversation", ""):
                    update_fields["title"] = title_candidate

                await db.conversations.update_one(
                    {"thread_id": thread_id},
                    {
                        "$push": {"history": turn},
                        "$set": update_fields,
                    },
                    upsert=True,
                )
                logger.info("Persisted turn to MongoDB for thread '%s'. Total turns updated.", thread_id)
                return await self.get_or_create_thread(thread_id, user_id)
            except Exception as exc:
                logger.warning("MongoDB save failed for thread '%s': %s. Using memory fallback.", thread_id, exc)

        # Memory fallback
        thread_state = await self.get_or_create_thread(thread_id, user_id)
        thread_state["history"].append(turn)
        thread_state["context"] = context
        thread_state["updated_at"] = now
        if thread_state.get("title") in ("New Chat", "New Conversation", ""):
            thread_state["title"] = title_candidate
        return thread_state

    async def list_threads(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List active conversation threads sorted by most recent activity."""
        db = self._get_db_safe()
        if db is not None:
            try:
                query = {"user_id": user_id} if user_id else {}
                cursor = db.conversations.find(query).sort("updated_at", -1)
                docs = await cursor.to_list(length=100)

                result = []
                for d in docs:
                    history = d.get("history", [])
                    result.append({
                        "thread_id": d["thread_id"],
                        "title": d.get("title") or "New Chat",
                        "turns_count": len(history),
                        "updated_at": d.get("updated_at"),
                    })
                return result
            except Exception as exc:
                logger.warning("MongoDB list_threads failed: %s. Using memory fallback.", exc)

        # Memory fallback
        result = []
        for tid, data in self._memory_fallback.items():
            if user_id and data.get("user_id") != user_id:
                continue
            history = data.get("history", [])
            result.append({
                "thread_id": tid,
                "title": data.get("title") or "New Chat",
                "turns_count": len(history),
                "updated_at": data.get("updated_at"),
            })
        return result

    async def delete_thread(self, thread_id: str) -> bool:
        """Permanently delete a conversation thread and all its history."""
        db = self._get_db_safe()
        deleted = False
        if db is not None:
            try:
                res = await db.conversations.delete_one({"thread_id": thread_id})
                deleted = res.deleted_count > 0
                logger.info("Deleted thread '%s' from MongoDB (deleted: %s)", thread_id, deleted)
            except Exception as exc:
                logger.warning("MongoDB delete_thread failed for '%s': %s", thread_id, exc)

        # Also purge from memory fallback
        if thread_id in self._memory_fallback:
            del self._memory_fallback[thread_id]
            deleted = True

        return deleted

    async def delete_message(self, thread_id: str, message_index: int) -> bool:
        """Delete a particular message turn by index from a thread's history."""
        db = self._get_db_safe()
        if db is not None:
            try:
                doc = await db.conversations.find_one({"thread_id": thread_id})
                if doc and "history" in doc:
                    history = doc["history"]
                    if 0 <= message_index < len(history):
                        history.pop(message_index)
                        await db.conversations.update_one(
                            {"thread_id": thread_id},
                            {
                                "$set": {
                                    "history": history,
                                    "updated_at": datetime.now(timezone.utc),
                                }
                            },
                        )
                        logger.info("Deleted message index %d from thread '%s' in MongoDB.", message_index, thread_id)
                        return True
            except Exception as exc:
                logger.warning("MongoDB delete_message failed for '%s': %s", thread_id, exc)

        # Memory fallback
        if thread_id in self._memory_fallback:
            history = self._memory_fallback[thread_id].get("history", [])
            if 0 <= message_index < len(history):
                history.pop(message_index)
                return True

        return False


_chat_history_instance: Optional[ChatHistoryService] = None


def get_chat_history_service() -> ChatHistoryService:
    """Singleton getter for the ChatHistoryService instance."""
    global _chat_history_instance
    if _chat_history_instance is None:
        _chat_history_instance = ChatHistoryService()
    return _chat_history_instance
