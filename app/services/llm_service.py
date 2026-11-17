"""LLM Service handling dynamic intent classification, entity extraction, and intelligent conversational response generation with Google Gemini."""

import os
import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple
from dotenv import load_dotenv

from app.config.chromadb import ChromaService

load_dotenv()

logger = logging.getLogger("ai_chat_api.llm_service")


class LLMService:
    """Service to interact with Google Gemini and manage conversation state."""

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.getenv("GEMINI_API_KEY", "").strip() or None
        self.model_name: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
        self.provider: str = os.getenv("LLM_PROVIDER", "gemini").lower().strip()
        
        # In-memory thread state storage: thread_id -> state dict
        self._threads: Dict[str, Dict[str, Any]] = {}

        # Vector database service (ChromaDB) for RAG
        self.chroma_service = ChromaService()
        
        # Priority list of models to try in case of temporary rate limits or 503 load
        self.candidate_models: List[str] = [
            self.model_name,
            "gemini-3.5-flash",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-3.8-flash",
        ]
        # Remove duplicates while preserving order
        self.candidate_models = list(dict.fromkeys(self.candidate_models))
        
        self._init_gemini_client()

    def _init_gemini_client(self) -> None:
        """Initialize Google GenAI client if credentials exist."""
        self.gemini_client = None
        if not self.api_key:
            logger.info("No GEMINI_API_KEY detected in environment. Using fallback engine.")
            return

        try:
            from google import genai
            self.gemini_client = genai.Client(api_key=self.api_key)
            logger.info("Google GenAI client initialized successfully with preferred model %s.", self.model_name)
        except Exception as exc:
            logger.warning("Could not initialize google.genai: %s", exc)
            self.gemini_client = None

    def get_thread_state(self, thread_id: str) -> Dict[str, Any]:
        """Retrieve or initialize conversation state for a given thread_id."""
        if thread_id not in self._threads:
            self._threads[thread_id] = {
                "thread_id": thread_id,
                "history": [],
                "awaiting": None,
                "context": {
                    "result": {},
                    "username": "Admin",
                },
            }
        return self._threads[thread_id]

    def list_all_threads(self) -> List[Dict[str, Any]]:
        """List all active conversation threads with summaries."""
        result = []
        for tid, data in self._threads.items():
            history = data.get("history", [])
            last_message = ""
            if history:
                last_turn = history[-1]
                last_message = last_turn.get("user", "")
            result.append({
                "thread_id": tid,
                "title": (last_message[:35] + "...") if len(last_message) > 35 else (last_message or "New Chat"),
                "turns_count": len(history),
            })
        return result

    def delete_message(self, thread_id: str, message_index: int) -> bool:
        """Delete a particular message turn by index from a thread's history."""
        if thread_id in self._threads:
            history = self._threads[thread_id].get("history", [])
            if 0 <= message_index < len(history):
                history.pop(message_index)
                return True
        return False

    def clear_thread_state(self, thread_id: str) -> bool:
        """Delete a particular chat thread and all its conversation memory."""
        if thread_id in self._threads:
            del self._threads[thread_id]
            return True
        return False

    async def generate_chat_turn(
        self,
        thread_id: str,
        user_text: str,
        channel: str = "rest",
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any], str, float, Dict[str, Any]]:
        """Process user message and generate bot_messages, entities, intent, confidence, and updated context."""
        state = self.get_thread_state(thread_id)

        # Merge provided context
        merged_context = dict(state.get("context", {}))
        if context:
            merged_context.update(context)
        username = merged_context.get("username", "Admin")

        # Query ChromaDB for semantic knowledge retrieval (RAG)
        retrieved_docs = self.chroma_service.query_similar(user_text, n_results=2)
        if retrieved_docs:
            merged_context["retrieved_context"] = [
                {"source": d["metadata"].get("source", "KB"), "title": d["metadata"].get("title", ""), "text": d["text"]}
                for d in retrieved_docs
            ]

        # 1. Try Live Gemini API if client is available
        if self.gemini_client and self.provider == "gemini":
            try:
                gemini_result = await self._call_gemini(
                    thread_id=thread_id,
                    user_text=user_text,
                    username=username,
                    state=state,
                    context=merged_context,
                    retrieved_docs=retrieved_docs,
                )
                if gemini_result:
                    bot_msgs, entities, intent, confidence, ctx = gemini_result
                    # Save to conversation history
                    state["history"].append({"user": user_text, "bot": [m["text"] for m in bot_msgs]})
                    state["context"] = ctx
                    return bot_msgs, entities, intent, confidence, ctx
            except Exception as exc:
                logger.warning("Gemini API call failed (%s), utilizing dynamic response engine.", exc)

        # 2. Dynamic Fallback Engine (Intelligently handles questions, orders, and chitchat)
        bot_msgs, entities, intent, confidence, ctx = self._dynamic_fallback_engine(
            thread_id=thread_id,
            user_text=user_text,
            username=username,
            state=state,
            context=merged_context,
            retrieved_docs=retrieved_docs,
        )

        state["history"].append({"user": user_text, "bot": [m["text"] for m in bot_msgs]})
        state["context"] = ctx
        return bot_msgs, entities, intent, confidence, ctx

    async def _call_gemini(
        self,
        thread_id: str,
        user_text: str,
        username: str,
        state: Dict[str, Any],
        context: Dict[str, Any],
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
    ) -> Optional[Tuple[List[Dict[str, str]], Dict[str, Any], str, float, Dict[str, Any]]]:
        """Invoke Google Gemini model with structured output prompt and multi-model fallback."""
        from google.genai import types

        history_summary = state.get("history", [])[-6:]  # Keep last 6 conversational turns
        
        kb_text = ""
        if retrieved_docs:
            kb_text = "\n".join([
                f"- [Source: {d['metadata'].get('source', 'KB')}] ({d['metadata'].get('title', '')}): {d['text']}"
                for d in retrieved_docs
            ])

        system_instruction = f"""You are an intelligent, conversational AI assistant communicating in real time with a human user in a chat interface and admin panel.
Logged-in User: {username}
Conversation History (previous turns): {json.dumps(history_summary)}
Current State Context: {json.dumps(context)}

VERIFIED KNOWLEDGE BASE CONTEXT (Retrieved from ChromaDB vector store):
{kb_text or "No specific knowledge base context was retrieved for this turn."}

INSTRUCTIONS:
1. Provide a direct, intelligent, and accurate answer to the user's specific question or statement. Do NOT give canned or static answers.
2. If the user asks a question about policies, procedures, products, shipping, returns, or support, use the verified ChromaDB context above when relevant and cite the source.
3. If the user asks about an order or shipment status:
   - If they provided an order number (e.g. ORD987124), extract it into entities.order_number, give confirmation, and provide status details with HTML tags like <b>Shipped</b> and estimated arrival.
   - If they did not provide an order number yet, politely ask for it.
4. If the user greets you, greet them warmly by their username.
5. Classify the user's intent accurately (e.g., question_answering, check_order_status, greeting, explanation_request, product_inquiry, general_chat).
6. Assign a realistic confidence score between 0.0 and 1.0.
7. Extract all relevant entities (topics, order numbers, dates, locations, products) as key-value pairs.
8. Output ONLY valid JSON matching this schema:
{{
  "intent": "<classified_intent>",
  "confidence": <confidence_float>,
  "entities": {{ "<entity_key>": "<entity_value>" }},
  "bot_message": [
    {{ "text": "<first_message_or_direct_response>" }},
    {{ "text": "<optional_second_message_or_details>" }}
  ],
  "context": {{
    "result": {{}},
    "username": "{username}"
  }}
}}
"""

        prompt = f"""User message: "{user_text}"\n\nGenerate the conversational response JSON:"""

        for model in self.candidate_models:
            try:
                response = self.gemini_client.models.generate_content(
                    model=model,
                    contents=system_instruction + "\n\n" + prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.7,
                    ),
                )
                
                response_text = response.text.strip() if response and response.text else ""
                if not response_text:
                    continue

                # Strip markdown fences if present
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                if response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

                data = json.loads(response_text)
                intent = data.get("intent", "question_answering")
                confidence = float(data.get("confidence", 0.90))
                entities = data.get("entities", {})
                
                raw_msgs = data.get("bot_message", [])
                if isinstance(raw_msgs, list) and len(raw_msgs) > 0:
                    bot_msgs = [
                        {"text": m.get("text", "")} if isinstance(m, dict) else {"text": str(m)}
                        for m in raw_msgs
                    ]
                else:
                    bot_msgs = [{"text": str(data.get("text", "I understand. How else can I assist you?"))}]

                res_context = data.get("context", {})
                if not isinstance(res_context, dict):
                    res_context = {}
                # Preserve merged context keys such as retrieved_context
                for k, v in context.items():
                    if k not in res_context:
                        res_context[k] = v
                if "username" not in res_context:
                    res_context["username"] = username
                if "result" not in res_context:
                    res_context["result"] = {}

                logger.info("Successfully generated dynamic response using Gemini model '%s'. Intent: %s", model, intent)
                return bot_msgs, entities, intent, confidence, res_context

            except Exception as model_err:
                logger.warning("Attempt with model '%s' failed: %s. Trying next candidate model...", model, model_err)
                continue

        return None

    def _dynamic_fallback_engine(
        self,
        thread_id: str,
        user_text: str,
        username: str,
        state: Dict[str, Any],
        context: Dict[str, Any],
        retrieved_docs: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any], str, float, Dict[str, Any]]:
        """Dynamic fallback responder that contextually answers questions using ChromaDB knowledge when available."""
        text_lower = user_text.lower().strip()
        entities: Dict[str, Any] = {}
        awaiting = state.get("awaiting")

        # 0. Check if ChromaDB returned relevant knowledge base context
        if retrieved_docs and len(retrieved_docs) > 0:
            top_hit = retrieved_docs[0]
            if any(k in text_lower for k in ["return", "refund", "shipping", "deliver", "hour", "contact", "policy", "standard"]):
                intent = "knowledge_retrieval"
                confidence = 0.94
                bot_messages = [
                    {"text": top_hit["text"]},
                    {"text": f"<i>(Source: {top_hit['metadata'].get('source', 'Knowledge Base')})</i>"}
                ]
                context["result"] = {"kb_source": top_hit["metadata"].get("source", "")}
                context["username"] = username
                return bot_messages, entities, intent, confidence, context

        # 1. Order Number detection
        order_match = re.search(r"\b(ORD[-_]?[0-9]+[A-Za-z0-9]*)\b", user_text, re.IGNORECASE)
        if not order_match:
            order_match = re.search(r"\b([0-9]{6,10})\b", user_text)

        order_number = order_match.group(1).upper() if order_match else None

        if order_number or (awaiting == "order_number" and re.search(r"[0-9]", user_text)):
            extracted_num = order_number or re.sub(r"[^A-Za-z0-9]", "", user_text.strip().split()[-1]).upper()
            entities["order_number"] = extracted_num
            intent = "check_order_status"
            confidence = 0.6912927868354805
            bot_messages = [
                {"text": f" Let me check the status of order {extracted_num}"},
                {"text": "\n Your order status is <b>Shipped</b> and is expected to arrive in 2-3 business days."},
            ]
            context["result"] = {"order_number": extracted_num, "status": "Shipped", "eta": "2-3 business days"}
            context["username"] = username
            state["awaiting"] = None
            return bot_messages, entities, intent, confidence, context

        # 2. Inquiries about order status
        if any(keyword in text_lower for keyword in ["order", "status", "track", "delivery", "shipping"]):
            intent = "check_order_status"
            confidence = 0.75
            bot_messages = [
                {"text": "Sure"},
                {"text": "Can you please give me your order number ?"},
            ]
            state["awaiting"] = "order_number"
            context["result"] = {}
            context["username"] = username
            return bot_messages, entities, intent, confidence, context

        # 3. Greetings
        if any(greet in text_lower for greet in ["hi", "hello", "hey", "good morning", "good afternoon"]):
            intent = "greeting"
            confidence = 0.985
            bot_messages = [
                {"text": f"Hi {username}"},
                {"text": "What can i do for you ?"},
            ]
            context["result"] = {}
            context["username"] = username
            return bot_messages, entities, intent, confidence, context

        # 4. Questions about AI identity & capabilities
        if any(q in text_lower for q in ["who are you", "what are you", "what can you do", "help me"]):
            intent = "bot_identity_inquiry"
            confidence = 0.92
            bot_messages = [
                {
                    "text": f"I am your AI assistant powered by Google Gemini! You can ask me any question, discuss ideas, or track order statuses (e.g. by asking about order ORD987124)."
                }
            ]
            context["result"] = {}
            context["username"] = username
            return bot_messages, entities, intent, confidence, context

        # 5. Question Answering detection (what, why, how, who, explain, tell me)
        if any(text_lower.startswith(q) for q in ["what", "why", "how", "who", "when", "where", "can you", "explain", "tell me"]):
            intent = "question_answering"
            confidence = 0.88
            entities["query"] = user_text
            bot_messages = [
                {
                    "text": f"Regarding your question about '{user_text}': I am ready to explore this with you! Connect your Gemini API key in .env for live multi-turn reasoning on any topic."
                }
            ]
            context["result"] = {}
            context["username"] = username
            return bot_messages, entities, intent, confidence, context

        # 6. General conversational reply tailored to user input
        intent = "general_conversation"
        confidence = 0.78
        bot_messages = [
            {
                "text": f"I received your message: \"{user_text}\". How would you like to proceed or explore this topic further?"
            }
        ]
        context["result"] = {}
        context["username"] = username
        return bot_messages, entities, intent, confidence, context
