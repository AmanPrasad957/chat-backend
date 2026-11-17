"""Chat request and response schemas matching the Admin Panel chat state specification."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, model_validator


class ChatRequest(BaseModel):
    """Incoming chat message request payload."""

    text: Optional[str] = Field(
        default=None,
        description="The user message text.",
        examples=["can you help to find the status of my order", "it's ORD987124"],
    )
    message: Optional[str] = Field(
        default=None,
        description="Alternative field for message text if text is omitted.",
    )
    thread_id: str = Field(
        default="test-user",
        description="Unique identifier for the user session or conversation thread.",
        examples=["test-user"],
    )
    channel: str = Field(
        default="rest",
        description="Channel through which the message was sent (e.g. rest, web, slack).",
        examples=["rest"],
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context passed by caller (e.g. username, auth tokens, session variables).",
        examples=[{"username": "Admin"}],
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_message_text(cls, values: Any) -> Any:
        """Allow either 'text' or 'message' to provide the user's input string."""
        if isinstance(values, dict):
            text_val = values.get("text")
            msg_val = values.get("message")
            if not text_val and msg_val:
                values["text"] = msg_val
            elif not text_val and not msg_val:
                raise ValueError("Either 'text' or 'message' must be provided in the request.")
        return values


class UserMessage(BaseModel):
    """User message structure as tracked in the chat state."""

    thread_id: str
    text: str
    channel: str = "rest"
    context: Dict[str, Any] = Field(default_factory=dict)


class BotMessageItem(BaseModel):
    """Individual message item from the bot response."""

    text: str


class IntentModel(BaseModel):
    """Classified intent with confidence score."""

    intent: str
    confidence: float


class NLUModel(BaseModel):
    """Natural Language Understanding output containing extracted entities and classified intent."""

    entities: Dict[str, Any] = Field(default_factory=dict)
    intent: IntentModel


class ChatResponse(BaseModel):
    """Complete chat response state matching the admin interface state schema."""

    thread_id: str
    user_message: UserMessage
    bot_message: List[BotMessageItem]
    nlu: NLUModel
    context: Dict[str, Any] = Field(default_factory=dict)
