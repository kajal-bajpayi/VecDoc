
import sys, argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ingestion import ingest_pdf
from pipeline import VectorStore, chunk_doc, answer, summarize, search

console = Console()
store   = VectorStore()


def cmd_ingest(path: str):
    console.print(f"\n[bold]Ingesting:[/bold] {path}")
    with console.status("Reading PDF…"):
        doc = ingest_pdf(path)
    if not doc["text"].strip():
        console.print("[red]Empty document — aborting.[/red]"); return

    for w in doc["warnings"]:
        console.print(f"  [yellow]⚠ {w}[/yellow]")

    with console.status("Chunking and embedding…"):
        chunks = chunk_doc(doc)
        n = store.add(chunks)
    console.print(f"  ✓ [bold]{doc['title']!r}[/bold] — {doc['metadata']['pages']} pages, {n} chunks stored\n")


def cmd_ask(query, title):
    console.print(f"\n[bold]Q:[/bold] {query}\n")
    with console.status("Retrieving…"):
        r = answer(query, store, doc_title=title)
    console.print(Panel(r["answer"], title="Answer", border_style="green"))
    if r["chunks"]:
        console.print(f"\n[dim]Used {len(r['chunks'])} chunks:[/dim]")
        for i, c in enumerate(r["chunks"], 1):
            console.print(f"  [{i}] {c['doc_title']!r}  chunk {c['chunk_index']}  score={round(1-c['distance'],3)}")
    print()


def cmd_summarize(title):
    scope = title or "all documents"
    console.print(f"\n[bold]Summarizing:[/bold] {scope}\n")
    with console.status("Summarizing…"):
        r = summarize(store, doc_title=title)
    console.print(Panel(r["summary"], title=f"Summary — {scope}", border_style="blue"))
    console.print(f"\n[dim]Based on {r['chunks_used']} chunks.[/dim]\n")


def cmd_search(query, title, top_k):
    console.print(f"\n[bold]Search:[/bold] {query!r}\n")
    results = search(query, store, top_k=top_k, doc_title=title)
    if not results:
        console.print("[yellow]No results.[/yellow]"); return
    for i, r in enumerate(results, 1):
        bar = "█" * int(r["score"]*20) + "░" * (20 - int(r["score"]*20))
        console.print(f"[cyan][{i}][/cyan] {r['doc_title']!r}  chunk {r['chunk_index']}  {r['score']:.3f}  {bar}")
        console.print(f"    [dim]{r['text'][:160].replace(chr(10),' ')}…[/dim]\n")


def cmd_sources():
    srcs = store.sources()
    if not srcs:
        console.print("[yellow]Nothing ingested yet.[/yellow]"); return
    t = Table(title=f"Sources ({len(srcs)})", show_lines=True)
    t.add_column("Title", style="bold"); t.add_column("File")
    for s in srcs:
        t.add_row(s["title"] or "—", s["source_ref"])
    console.print(t)
    console.print(f"\n[dim]Total chunks: {store.count()}[/dim]\n")


def cmd_delete(ref):
    n = store.delete(ref)
    msg = f"Deleted {n} chunks for {ref!r}" if n else f"Nothing found for {ref!r}"
    console.print(f"[{'green' if n else 'yellow'}]{msg}[/]")


def main():
    p = argparse.ArgumentParser(description="PDF RAG")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("ingest").add_argument("path")

    pa = sub.add_parser("ask")
    pa.add_argument("query"); pa.add_argument("--title")

    ps = sub.add_parser("summarize")
    ps.add_argument("--title")

    psr = sub.add_parser("search")
    psr.add_argument("query"); psr.add_argument("--title")
    psr.add_argument("--top-k", type=int, default=6)

    sub.add_parser("sources")

    pd = sub.add_parser("delete")
    pd.add_argument("ref")

    args = p.parse_args()
    try:
        if   args.cmd == "ingest":    cmd_ingest(args.path)
        elif args.cmd == "ask":       cmd_ask(args.query, args.title)
        elif args.cmd == "summarize": cmd_summarize(args.title)
        elif args.cmd == "search":    cmd_search(args.query, args.title, args.top_k)
        elif args.cmd == "sources":   cmd_sources()
        elif args.cmd == "delete":    cmd_delete(args.ref)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
    except Exception as e:
        console.print(f"\n[red]Error:[/red] {e}"); sys.exit(1)


if __name__ == "__main__":
    main()