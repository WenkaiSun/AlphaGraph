from __future__ import annotations
from typing import Any, Dict, List
from transformers import pipeline
import openai


class Planner:
    """Node that generates a search plan from the user query."""
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        # Simple plan generation - can be enhanced with LLM
        state["plan"] = f"Search for information about: {query}"
        return state


class Retriever:
    """Node that retrieves relevant documents from the vector store."""
    
    def __init__(self, vs, top_k: int = 8, bm25_boost: float = 0.2):
        self.vs = vs
        self.top_k = top_k
        self.bm25_boost = bm25_boost
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        query = state.get("query", "")
        docs = self.vs.search(query, top_k=self.top_k, bm25_boost=self.bm25_boost)
        state["docs"] = docs
        state["context"] = "\n\n".join([d.get("text", "") for d in docs])
        return state


class Synthesizer:
    """Node that synthesizes retrieved documents into a summary."""
    
    def __init__(self, mode: str = "local", openai_model: str = "gpt-4o-mini"):
        self.mode = mode
        self.model = openai_model
        if mode == "openai":
            self.client = openai.OpenAI()
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        context = state.get("context", "")
        query = state.get("query", "")
        
        if self.mode == "openai":
            summary = self._openai_summarize(context, query)
        else:
            summary = self._local_summarize(context)
        
        state["summary"] = summary
        return state
    
    def _openai_summarize(self, context: str, query: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst. Summarize the following context in relation to the query."},
                    {"role": "user", "content": f"Query: {query}\n\nContext:\n{context[:4000]}"}
                ],
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating summary: {str(e)}"
    
    def _local_summarize(self, context: str) -> str:
        # Simple extractive summary - take first 500 chars
        return context[:500] + ("..." if len(context) > 500 else "")


class FinancialNER:
    """Node that extracts financial entities using OpenAI function calling."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        self.client = openai.OpenAI()
    
    def _fallback(self, text: str) -> List[Dict[str, Any]]:
        """Simple fallback entity extraction using regex."""
        import re
        entities = []
        # Simple ticker pattern (1-5 uppercase letters)
        tickers = re.findall(r'\b[A-Z]{1,5}\b', text)
        for ticker in set(tickers):
            entities.append({
                "type": "ticker",
                "value": ticker,
                "evidence": ""
            })
        return entities
    
    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        payload = state.get("summary", "")
        
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "record_entities",
                    "description": "Record extracted financial entities",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entities": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "type": {"type": "string", "enum": ["ticker", "company", "metric", "date"]},
                                        "value": {"type": "string"},
                                        "evidence": {"type": "string"}
                                    },
                                    "required": ["type", "value"]
                                }
                            }
                        },
                        "required": ["entities"]
                    }
                }
            }
        ]
        
        system = (
            "You are a precise Financial NER engine. Extract entities relevant to equities and macro."
            " Prefer official tickers (e.g., MSFT), capture numeric values and units, and include short evidence spans."
        )
        user = f"Extract entities from the following text:\n\n{payload}"
        
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "record_entities"}}
            )
            
            entities: List[Dict[str, Any]] = []
            for choice in resp.choices:
                if choice.message.tool_calls:
                    for tool_call in choice.message.tool_calls:
                        if tool_call.function.name == "record_entities":
                            import json
                            args = json.loads(tool_call.function.arguments)
                            entities.extend(args.get("entities", []))
            
            state["entities"] = entities if entities else self._fallback(payload)
        except Exception:
            state["entities"] = self._fallback(payload)
        
        return state


class SignalExtractor:
    """Node that extracts trading signals from entities."""
    
    def __init__(self, min_strength: float = 0.3, whitelist: List[str] | None = None):
        self.min_strength = min_strength
        self.whitelist = set(whitelist or [])
        try:
            self.sent_clf = pipeline("sentiment-analysis")
        except Exception:
            self.sent_clf = None

    def _sentiment(self, text: str) -> float:
        if not self.sent_clf:
            return 0.0
        res = self.sent_clf(text[:512])[0]
        if res["label"].lower().startswith("pos"):
            return float(res["score"])  # 0..1 positive
        if res["label"].lower().startswith("neg"):
            return -float(res["score"])  # -1..0 negative
        return 0.0

    def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        ents = state.get("entities", [])
        signals: List[Dict[str, Any]] = []
        
        for e in ents:
            if e.get("type") == "ticker":
                t = e.get("value")
                if self.whitelist and t not in self.whitelist:
                    continue
                evs = [x.get("evidence", "") for x in ents if x.get("evidence") and (t in x.get("evidence", ""))]
                text = max(evs, key=len) if evs else state.get("summary", "")
                score = self._sentiment(text)
                if abs(score) >= self.min_strength:
                    signals.append({
                        "ticker": t,
                        "sentiment": score,
                        "evidence": text[:300],
                    })
        
        state["signals"] = sorted(signals, key=lambda s: -abs(s["sentiment"]))
        return state
