from __future__ import annotations
@dataclass
class DocChunk:
    doc_id: str
    chunk_id: int
    text: str
    meta: dict




class Ingestor:
    def __init__(self, chunk_size: int = 1200, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap


def load_dir(self, data_dir: str) -> List[DocChunk]:
    paths = list(Path(data_dir).glob("**/*"))
    chunks: List[DocChunk] = []
    for p in paths:
        if p.is_dir():
            continue
        text = self._read_file(p)
        if not text:
            continue
        chunks.extend(self._chunk_text(text, doc_id=str(p)))
        return chunks


def _read_file(self, p: Path) -> str:
    try:
    if p.suffix.lower() in {".txt", ".md"}:
    return p.read_text(errors="ignore")
    if p.suffix.lower() in {".html", ".htm"}:
    soup = BeautifulSoup(p.read_text(errors="ignore"), "html.parser")
    return soup.get_text(" ")
    if p.suffix.lower() in {".pdf"}:
    return pdf_extract(str(p))
    # Fallback: try unstructured plain text partition
    parts = partition_text(filename=str(p))
    return "\n".join([el.text for el in parts if getattr(el, "text", None)])
    except Exception:
    return ""


def _chunk_text(self, text: str, doc_id: str) -> List[DocChunk]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: List[DocChunk] = []
    i = 0
    start = 0
    while start < len(text):
        end = min(len(text), start + self.chunk_size)
        chunk = text[start:end]
        chunks.append(DocChunk(doc_id=doc_id, chunk_id=i, text=chunk, meta={"source": doc_id}))
        i += 1
        start = end - self.overlap
        if start < 0:
            start = 0
    return chunks
