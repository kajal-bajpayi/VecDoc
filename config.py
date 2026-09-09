from dataclasses import dataclass, field
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

@dataclass
class Config:
    anthropic_api_key: str = field(default_factory=lambda: os.environ["ANTHROPIC_API_KEY"])
    claude_model: str      = "claude-sonnet-4-6"
    embedding_model: str   = "all-MiniLM-L6-v2"
    chroma_dir: Path       = Path(".chroma")
    collection_name: str   = "pdf_rag"
    chunk_size: int        = 512
    chunk_overlap: int     = 64
    top_k: int             = 6

cfg = Config()