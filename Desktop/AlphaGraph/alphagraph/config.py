from __future__ import annotations
from pydantic_settings import BaseSettings
from pydantic import BaseModel
from typing import List
import yaml
import os


class ChunkConfig(BaseModel):
size: int = 1200
overlap: int = 200


class RetrieveConfig(BaseModel):
top_k: int = 8
bm25_boost: float = 0.2


class AlphaConfig(BaseModel):
min_sentiment_strength: float = 0.3
ticker_whitelist: List[str] = []


class Config(BaseSettings):
embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
summary_model: str = "local" # local | openai
openai_model: str = "gpt-4o-mini"
chunk: ChunkConfig = ChunkConfig()
retrieve: RetrieveConfig = RetrieveConfig()
alpha: AlphaConfig = AlphaConfig()




def load_config(path: str | None) -> Config:
if path and os.path.exists(path):
with open(path, "r") as f:
data = yaml.safe_load(f) or {}
return Config(**data)
return Config()
