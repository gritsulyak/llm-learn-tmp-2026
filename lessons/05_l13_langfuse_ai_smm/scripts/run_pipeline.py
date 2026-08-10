import argparse
import json
import sys
import time
from pathlib import Path

from src.config import settings
from src.graph.build_graph import build_graph
from src.observability.langfuse_setup import (
    get_langfuse_handler,
    shutdown_tracing,
    trace_config,
)


def run(niche: str, social_network: str, max_revision_rounds: int) -> dict:
    graph = build_graph()

    handler = get_langfuse_handler()
    config = trace_config(niche, social_network)
    if handler is not None:
        config["callbacks"] = [handler]

    state = {
        "niche": niche,
        "social_network": social_network,
        "content_plan": [],
        "posts_to_write": [],
        "current_post_index": 0,
        "draft_text": "",
        "editor_verdict": "revise",
        "editor_comments": "",
        "revision_round": 0,
        "max_revision_rounds": max_revision_rounds,
        "finished_posts": [],
        "phase": "planning",
        "error": None,
    }

    started = time.perf_counter()
    result = graph.invoke(state, config=config)
    elapsed = time.perf_counter() - started

    print("\n--- done ---")
    print(f"elapsed: {elapsed:.1f}s")
    print(f"content_plan items: {len(result.get('content_plan', []))}")
    print(f"finished posts: {len(result.get('finished_posts', []))}")

    shutdown_tracing()
    return result


def save_result(result: dict, output_dir: str) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = out_dir / f"run_{ts}.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {path}")
    return path


def build_argparser() -> argparse.ArgumentParser:
    cfg = settings()
    p = argparse.ArgumentParser(description="Run AI SMM multi-agent pipeline locally (Ollama).")
    p.add_argument("--niche", default=cfg.niche)
    p.add_argument("--social-network", default=cfg.social_network)
    p.add_argument("--max-revision-rounds", type=int, default=cfg.max_revision_rounds)
    p.add_argument("--output-dir", default=cfg.output_dir)
    p.add_argument("--graph", action="store_true", help="Print mermaid graph and exit.")
    return p


if __name__ == "__main__":
    args = build_argparser().parse_args()

    if args.graph:
        print(build_graph().get_graph().draw_mermaid())
        sys.exit(0)

    result = run(args.niche, args.social_network, args.max_revision_rounds)
    save_result(result, args.output_dir)
