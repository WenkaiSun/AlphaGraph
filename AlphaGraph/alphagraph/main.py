import argparse
from rich import print
from alphagraph.config import load_config
from alphagraph.ingest import Ingestor
from alphagraph.store import VectorStore
from alphagraph.graph import AlphaGraphRunner


def do_index(args):
    cfg = load_config(args.config)
    ing = Ingestor(chunk_size=cfg.chunk.size, overlap=cfg.chunk.overlap)
    chunks = ing.load_dir(args.data_dir)
    print(f"[bold green]Loaded[/] {len(chunks)} chunks from {args.data_dir}")
    vs = VectorStore(index_dir=args.index_dir, embedding_model=cfg.embedding_model)
    vs.build(chunks)
    print(f"[bold green]Built index[/] at {args.index_dir}")


def do_query(args):
    cfg = load_config(args.config)
    vs = VectorStore(index_dir=args.index_dir, embedding_model=cfg.embedding_model).load()
    runner = AlphaGraphRunner(
        vs,
        top_k=cfg.retrieve.top_k,
        bm25_boost=cfg.retrieve.bm25_boost,
        summary_mode=cfg.summary_model,
        openai_model=cfg.openai_model,
        min_strength=cfg.alpha.min_sentiment_strength,
        whitelist=cfg.alpha.ticker_whitelist,
    )
    out = runner.run(args.query)
    print("\n[bold]Plan:[/]", out.get("plan"))
    print("\n[bold]Top Docs:[/]")
    for i, d in enumerate(out.get("docs", []), 1):
        print(f"[dim]{i:2d}[/] score={d['score']:.3f} src={d['meta'].get('source','-')}\n " + d["text"][:180].replace("\n"," ") + ("..." if len(d["text"])>180 else ""))
    print("\n[bold]Summary:[/]\n", out.get("summary", ""))
    print("\n[bold]Signals:[/]")
    for s in out.get("signals", []):
        sentiment = "bullish" if s["sentiment"]>0 else "bearish"
        print(f"- {s['ticker']}: {sentiment} ({s['sentiment']:+.2f})\n  evidence: {s['evidence']}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("query", nargs="?")
    p.add_argument("--mode", choices=["index", "query"], default="query")
    p.add_argument("--data_dir", default="./data")
    p.add_argument("--index_dir", default="./index")
    p.add_argument("--config", default="config.yaml")
    args = p.parse_args()

    if args.mode == "index":
        return do_index(args)
    if not args.query:
        p.error("query text is required in --mode query")
    return do_query(args)


if __name__ == "__main__":
    main()
