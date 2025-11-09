from __future__ import annotations


class AlphaGraphRunner:
    def __init__(self, vs: VectorStore, *, top_k: int, bm25_boost: float, summary_mode: str, openai_model: str, min_strength: float, whitelist: List[str]):
        self.vs = vs
        self.graph = StateGraph(GraphState)
        self.graph.add_node("plan", Planner())
        self.graph.add_node("retrieve", Retriever(vs, top_k=top_k, bm25_boost=bm25_boost))
        self.graph.add_node("synthesize", Synthesizer(mode=summary_mode, openai_model=openai_model))
        self.graph.add_node("ner", FinancialNER(openai_model))
        self.graph.add_node("signals", SignalExtractor(min_strength=min_strength, whitelist=whitelist))


        self.graph.set_entry_point("plan")
        self.graph.add_edge("plan", "retrieve")
        self.graph.add_edge("retrieve", "synthesize")
        self.graph.add_edge("synthesize", "ner")
        self.graph.add_edge("ner", "signals")
        self.graph.add_edge("signals", END)


        self._app = self.graph.compile()


    def run(self, query: str) -> Dict[str, Any]:
        init: GraphState = {"query": query, "plan": "", "docs": [], "context": "", "summary": "", "entities": [], "signals": []}
        out = self._app.invoke(init)
        return out


from __future__ import annotations
from typing import Any, Dict, TypedDict, List
from langgraph.graph import StateGraph, END
from .nodes import Planner, Retriever, Synthesizer, SignalExtractor
from .store import VectorStore


class GraphState(TypedDict):
    query: str
    plan: str
    docs: List[Dict[str, Any]]
    context: str
    summary: str
    signals: List[Dict[str, Any]]


class AlphaGraphRunner:
    def __init__(self, vs: VectorStore, *, top_k: int, bm25_boost: float, summary_mode: str, openai_model: str, min_strength: float, whitelist: List[str]):
        self.vs = vs
        self.graph = StateGraph(GraphState)
        self.graph.add_node("plan", Planner())
        self.graph.add_node("retrieve", Retriever(vs, top_k=top_k, bm25_boost=bm25_boost))
        self.graph.add_node("synthesize", Synthesizer(mode=summary_mode, openai_model=openai_model))
        self.graph.add_node("signals", SignalExtractor(min_strength=min_strength, whitelist=whitelist))


        self.graph.set_entry_point("plan")
        self.graph.add_edge("plan", "retrieve")
        self.graph.add_edge("retrieve", "synthesize")
        self.graph.add_edge("synthesize", "signals")
        self.graph.add_edge("signals", END)


        self._app = self.graph.compile()


    def run(self, query: str) -> Dict[str, Any]:
        init: GraphState = {"query": query, "plan": "", "docs": [], "context": "", "summary": "", "signals": []}
        out = self._app.invoke(init)
        return out
