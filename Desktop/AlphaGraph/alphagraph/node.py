from __future__ import annotations
}
}
]
system = (
"You are a precise Financial NER engine. Extract entities relevant to equities and macro."
" Prefer official tickers (e.g., MSFT), capture numeric values and units, and include short evidence spans."
)
user = f"Extract entities from the following text.


{payload}"
resp = self.client.responses.create(
model=self.model,
input=[{"role":"system","content":system},{"role":"user","content":user}],
tools=tools,
tool_choice={"type":"function","function":{"name":"record_entities"}}
)
entities: List[Dict[str,Any]] = []
for item in resp.output or []:
if item.type == "tool_call" and getattr(item, "name", "") == "record_entities":
args = item.arguments or {}
entities.extend(args.get("entities", []))
state["entities"] = entities if entities else self._fallback(payload)
return state


class SignalExtractor:
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
return float(res["score"]) # 0..1 positive
if res["label"].lower().startswith("neg"):
return -float(res["score"]) # -1..0 negative
return 0.0


def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
ents = state.get("entities", [])
signals: List[Signal] = []
for e in ents:
if e.get("type") == "ticker":
t = e.get("value")
if self.whitelist and t not in self.whitelist:
continue
evs = [x.get("evidence","") for x in ents if x.get("evidence") and (t in x.get("evidence",""))]
text = max(evs, key=len) if evs else state.get("summary","")
score = self._sentiment(text)
if abs(score) >= self.min_strength:
signals.append({
"ticker": t,
"sentiment": score,
"evidence": text[:300],
})
state["signals"] = sorted(signals, key=lambda s: -abs(s["sentiment"]))
return state
