"""LLM Service handling dynamic intent classification, entity extraction, and intelligent conversational response generation with Google Gemini."""

import os
import re
import json
import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple
from dotenv import load_dotenv

from app.config.chromadb import ChromaService
from app.services.chat_history_service import get_chat_history_service, ChatHistoryService

load_dotenv()

logger = logging.getLogger("ai_chat_api.llm_service")


class LLMService:
    """Service to interact with Google Gemini and manage conversation state."""

    def __init__(self) -> None:
        self.api_key: Optional[str] = os.getenv("GEMINI_API_KEY", "").strip() or None
        self.model_name: str = os.getenv("GEMINI_MODEL", "gemini-3.5-flash").strip()
        self.provider: str = os.getenv("LLM_PROVIDER", "gemini").lower().strip()

        # Chat history persistence service backed by MongoDB
        self.chat_history_service: ChatHistoryService = get_chat_history_service()

        # Vector database service (ChromaDB) for RAG
        self.chroma_service = ChromaService()
        
        # Active model to query
        self.candidate_models: List[str] = [self.model_name or "gemini-3.5-flash"]
        
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

    async def get_thread_state(self, thread_id: str, user_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve or initialize conversation state from MongoDB for a given thread_id."""
        return await self.chat_history_service.get_or_create_thread(thread_id, user_id)

    async def list_all_threads(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all active conversation threads from MongoDB."""
        return await self.chat_history_service.list_threads(user_id)

    async def delete_message(self, thread_id: str, message_index: int) -> bool:
        """Delete a particular message turn by index from MongoDB."""
        return await self.chat_history_service.delete_message(thread_id, message_index)

    async def clear_thread_state(self, thread_id: str) -> bool:
        """Delete a particular chat thread and all its conversation memory from MongoDB."""
        return await self.chat_history_service.delete_thread(thread_id)

    async def generate_chat_turn(
        self,
        thread_id: str,
        user_text: str,
        channel: str = "rest",
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> Tuple[List[Dict[str, str]], Dict[str, Any], str, float, Dict[str, Any]]:
        """Process user message and generate bot_messages, entities, intent, confidence, and updated context."""
        state = await self.get_thread_state(thread_id, user_id)

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
                    # Persist turn to MongoDB
                    await self.chat_history_service.save_turn(
                        thread_id=thread_id,
                        user_text=user_text,
                        bot_messages=bot_msgs,
                        intent=intent,
                        entities=entities,
                        context=ctx,
                        user_id=user_id,
                    )
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

        # Persist turn to MongoDB
        await self.chat_history_service.save_turn(
            thread_id=thread_id,
            user_text=user_text,
            bot_messages=bot_msgs,
            intent=intent,
            entities=entities,
            context=ctx,
            user_id=user_id,
        )
        return bot_msgs, entities, intent, confidence, ctx

    async def generate_chat_turn_stream(
        self,
        thread_id: str,
        user_text: str,
        channel: str = "rest",
        context: Optional[Dict[str, Any]] = None,
        user_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Process user message and yield SSE events ('start', 'token', 'done').

        Persists completed turn to MongoDB upon stream finish.
        """
        state = await self.get_thread_state(thread_id, user_id)

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

        # 1. Yield start event
        yield f"data: {json.dumps({'type': 'start', 'thread_id': thread_id})}\n\n"

        accumulated_text = ""
        gemini_streamed_success = False

        # 2. Try streaming from Gemini if client available
        if self.gemini_client and self.provider == "gemini":
            from google.genai import types

            history_summary = state.get("history", [])[-6:]
            kb_text = ""
            if retrieved_docs:
                kb_text = "\n".join([
                    f"- [Source: {d['metadata'].get('source', 'KB')}] ({d['metadata'].get('title', '')}): {d['text']}"
                    for d in retrieved_docs
                ])

            system_instruction = f"""You are an intelligent, conversational AI assistant communicating in real time with a human user.
Logged-in User: {username}
Conversation History (previous turns): {json.dumps(history_summary, default=str)}
Current State Context: {json.dumps(context or {}, default=str)}

VERIFIED KNOWLEDGE BASE CONTEXT (Retrieved from ChromaDB vector store):
{kb_text or "No specific knowledge base context was retrieved for this turn."}

INSTRUCTIONS:
1. LANGUAGE POLICY: This assistant exclusively supports English and Hindi (हिन्दी). All other languages are excluded. If the user writes or speaks in Hindi (Devanagari script or Hinglish), respond in natural Hindi using Devanagari script. If the user writes or speaks in English, respond in English. Match the user's language choice naturally.
2. Provide a direct, intelligent, and accurate conversational response to the user.
3. If order status is asked and an order number is given (e.g. ORD987124), state that it is Shipped and arriving in 2-3 business days. If not given, ask for order number.
4. Use verified ChromaDB context when answering questions about policies or shipping.
5. Output your answer directly as clear conversational text (do not output JSON).
"""
            prompt = f"""User message: "{user_text}"\n\nAssistant:"""

            for model in self.candidate_models:
                try:
                    logger.info("Attempting async streaming with Gemini model '%s'...", model)
                    stream = await self.gemini_client.aio.models.generate_content_stream(
                        model=model,
                        contents=system_instruction + "\n\n" + prompt,
                        config=types.GenerateContentConfig(
                            temperature=0.7,
                        ),
                    )

                    # Await the first chunk with a strict 3.5s timeout to guarantee zero hang
                    first_chunk = await asyncio.wait_for(stream.__anext__(), timeout=3.5)
                    if first_chunk and first_chunk.text:
                        accumulated_text += first_chunk.text
                        yield f"data: {json.dumps({'type': 'token', 'delta': first_chunk.text})}\n\n"
                        await asyncio.sleep(0.001)

                    async for chunk in stream:
                        if chunk and chunk.text:
                            accumulated_text += chunk.text
                            yield f"data: {json.dumps({'type': 'token', 'delta': chunk.text})}\n\n"
                            await asyncio.sleep(0.001)

                    if accumulated_text.strip():
                        gemini_streamed_success = True
                        logger.info("Successfully streamed Gemini response using model '%s'", model)
                        break
                except (asyncio.TimeoutError, StopAsyncIteration):
                    logger.warning("Streaming attempt with model '%s' timed out (>3.5s). Switching...", model)
                    continue
                except Exception as stream_err:
                    logger.warning("Streaming attempt with model '%s' failed: %s. Trying next...", model, stream_err)
                    continue

        # 3. Fallback streaming if Gemini was not available or failed
        if not gemini_streamed_success:
            bot_msgs, entities, intent, confidence, ctx = self._dynamic_fallback_engine(
                thread_id=thread_id,
                user_text=user_text,
                username=username,
                state=state,
                context=merged_context,
                retrieved_docs=retrieved_docs,
            )
            full_reply = " ".join([m.get("text", "") for m in bot_msgs])
            accumulated_text = full_reply

            # Stream words with smooth typing cadence
            words = re.findall(r"\S+|\s+", full_reply)
            for w in words:
                yield f"data: {json.dumps({'type': 'token', 'delta': w})}\n\n"
                await asyncio.sleep(0.015)
        else:
            bot_msgs = [{"text": accumulated_text}]
            _, entities, intent, confidence, ctx = self._dynamic_fallback_engine(
                thread_id=thread_id,
                user_text=user_text,
                username=username,
                state=state,
                context=merged_context,
                retrieved_docs=retrieved_docs,
            )

        # 4. Persist completed turn to MongoDB
        await self.chat_history_service.save_turn(
            thread_id=thread_id,
            user_text=user_text,
            bot_messages=bot_msgs,
            intent=intent,
            entities=entities,
            context=ctx,
            user_id=user_id,
        )

        # 5. Yield done event
        done_payload = {
            "type": "done",
            "thread_id": thread_id,
            "user_message": {
                "thread_id": thread_id,
                "text": user_text,
                "channel": channel,
                "context": context or {},
            },
            "bot_message": bot_msgs,
            "nlu": {
                "entities": entities,
                "intent": {"intent": intent, "confidence": confidence},
            },
            "context": ctx,
        }
        yield f"data: {json.dumps(done_payload, default=str)}\n\n"

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
Conversation History (previous turns): {json.dumps(history_summary, default=str)}
Current State Context: {json.dumps(context, default=str)}

VERIFIED KNOWLEDGE BASE CONTEXT (Retrieved from ChromaDB vector store):
{kb_text or "No specific knowledge base context was retrieved for this turn."}

INSTRUCTIONS:
1. LANGUAGE POLICY: This assistant exclusively supports English and Hindi (हिन्दी). All other languages are excluded. If the user writes or speaks in Hindi (Devanagari script or Hinglish), respond in natural Hindi using Devanagari script. If the user writes or speaks in English, respond in English. Match the user's language choice naturally.
2. Provide a direct, intelligent, and accurate answer to the user's specific question or statement. Do NOT give canned or static answers.
3. If the user asks a question about policies, procedures, products, shipping, returns, or support, use the verified ChromaDB context above when relevant and cite the source.
4. If the user asks about an order or shipment status:
   - If they provided an order number (e.g. ORD987124), extract it into entities.order_number, give confirmation, and provide status details with HTML tags like <b>Shipped</b> and estimated arrival.
   - If they did not provide an order number yet, politely ask for it.
5. If the user greets you, greet them warmly by their username.
6. Classify the user's intent accurately (e.g., question_answering, check_order_status, greeting, explanation_request, product_inquiry, general_chat).
7. Assign a realistic confidence score between 0.0 and 1.0.
8. Extract all relevant entities (topics, order numbers, dates, locations, products) as key-value pairs.
9. Output ONLY valid JSON matching this schema:
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
                response = await asyncio.wait_for(
                    self.gemini_client.aio.models.generate_content(
                        model=model,
                        contents=system_instruction + "\n\n" + prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            temperature=0.7,
                        ),
                    ),
                    timeout=3.5,
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
        is_hindi = bool(re.search(r"[\u0900-\u097F]", user_text)) or any(
            w in text_lower for w in ["namaste", "pranam", "kya haal", "kaise ho", "dhanyawad", "shukriya"]
        )

        # 0. Check if ChromaDB returned relevant knowledge base context
        if retrieved_docs and len(retrieved_docs) > 0:
            top_hit = retrieved_docs[0]
            if any(k in text_lower for k in ["return", "refund", "shipping", "deliver", "hour", "contact", "policy", "standard"]) or any(k in user_text for k in ["वापसी", "रिफंड", "शिपिंग", "डिलीवरी", "नीति"]):
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
            if is_hindi:
                bot_messages = [
                    {"text": f"ऑर्डर {extracted_num} की स्थिति जाँची जा रही है..."},
                    {"text": f"\nआपके ऑर्डर की स्थिति <b>Shipped</b> है और यह 2-3 कार्यदिवसों में पहुँच जाएगा।"},
                ]
            else:
                bot_messages = [
                    {"text": f" Let me check the status of order {extracted_num}"},
                    {"text": "\n Your order status is <b>Shipped</b> and is expected to arrive in 2-3 business days."},
                ]
            context["result"] = {"order_number": extracted_num, "status": "Shipped", "eta": "2-3 business days"}
            context["username"] = username
            state["awaiting"] = None
            return bot_messages, entities, intent, confidence, context

        # 2. Inquiries about order status
        if any(keyword in text_lower for keyword in ["order", "status", "track", "delivery", "shipping"]) or any(keyword in user_text for keyword in ["ऑर्डर", "स्थिति", "डिलीवरी", "पार्सल"]):
            intent = "check_order_status"
            confidence = 0.75
            if is_hindi:
                bot_messages = [
                    {"text": "ज़रूर!"},
                    {"text": "कृपया मुझे अपना ऑर्डर नंबर (Order Number) प्रदान करें?"},
                ]
            else:
                bot_messages = [
                    {"text": "Sure"},
                    {"text": "Can you please give me your order number ?"},
                ]
            state["awaiting"] = "order_number"
            context["result"] = {}
            context["username"] = username
            return bot_messages, entities, intent, confidence, context

        # 3. Greetings
        if any(greet in text_lower for greet in ["hi", "hello", "hey", "good morning", "good afternoon"]) or any(
            hg in user_text for hg in ["नमस्ते", "प्रणाम", "नमस्कार", "हेलो", "हाय"]
        ) or any(hg in text_lower for hg in ["namaste", "pranam", "kya haal", "kaise ho"]):
            intent = "greeting"
            confidence = 0.985
            if is_hindi:
                bot_messages = [
                    {"text": f"नमस्ते {username}!"},
                    {"text": "मैं आपकी क्या सहायता कर सकता हूँ?"},
                ]
            else:
                bot_messages = [
                    {"text": f"Hi {username}"},
                    {"text": "What can i do for you ?"},
                ]
            context["result"] = {}
            context["username"] = username
            return bot_messages, entities, intent, confidence, context

        # 4. Questions about AI identity & capabilities
        if any(q in text_lower for q in ["who are you", "what are you", "what can you do", "help me"]) or any(
            q in user_text for q in ["आप कौन हैं", "तुम कौन हो", "क्या कर सकते हो", "मदद करो"]
        ):
            intent = "bot_identity_inquiry"
            confidence = 0.92
            if is_hindi:
                bot_messages = [
                    {
                        "text": f"मैं Google Gemini द्वारा संचालित आपका AI सहायक हूँ! आप मुझसे हिंदी या English में कोई भी प्रश्न पूछ सकते हैं, ज्ञानकोष खोज सकते हैं, या अपने ऑर्डर की स्थिति (जैसे ORD987124) जान सकते हैं।"
                    }
                ]
            else:
                bot_messages = [
                    {
                        "text": f"I am your AI assistant powered by Google Gemini! You can ask me any question, discuss ideas, or track order statuses (e.g. by asking about order ORD987124)."
                    }
                ]
            context["result"] = {}
            context["username"] = username
            return bot_messages, entities, intent, confidence, context

        # 5. Question Answering detection (what, why, how, who, explain, tell me)
        if any(text_lower.startswith(q) for q in ["what", "why", "how", "who", "when", "where", "can you", "explain", "tell me"]) or any(
            user_text.startswith(q) for q in ["क्या", "क्यों", "कैसे", "कहाँ", "कब", "बताइए", "समझाएं"]
        ):
            intent = "question_answering"
            confidence = 0.88
            entities["query"] = user_text
            if is_hindi:
                bot_messages = [
                    {
                        "text": f"आपके प्रश्न '{user_text}' के संबंध में: मैं आपकी सहायता के लिए तैयार हूँ! .env में Gemini API Key जोड़कर लाइव मल्टी-टर्न AI उत्तर प्राप्त करें।"
                    }
                ]
            else:
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
        if is_hindi:
            bot_messages = [
                {
                    "text": f"मुझे आपका संदेश प्राप्त हुआ: \"{user_text}\"। मैं इसमें आपकी किस प्रकार सहायता कर सकता हूँ?"
                }
            ]
        else:
            bot_messages = [
                {
                    "text": f"I received your message: \"{user_text}\". How would you like to proceed or explore this topic further?"
                }
            ]
        context["result"] = {}
        context["username"] = username
        return bot_messages, entities, intent, confidence, context
