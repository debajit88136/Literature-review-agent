"""Manual smoke test: run a REAL arXiv search from the command line.

Usage:
    python try_search.py "LLM hallucination detection"
    python try_search.py "graph neural networks" -n 10
"""

import argparse
import logging

from lit_review.arxiv_client import ArxivError, search_arxiv


def main() -> None:
    parser = argparse.ArgumentParser(description="Search arXiv from the terminal.")
    parser.add_argument("topic", help="research topic, e.g. 'LLM hallucination detection'")
    parser.add_argument("-n", "--num", type=int, default=5, help="number of papers")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    try:
        papers = search_arxiv(args.topic, max_results=args.num)
    except (ValueError, ArxivError) as exc:
        print(f"Search failed: {exc}")
        raise SystemExit(1)

    if not papers:
        print("No papers found. Try broader keywords.")
        return

    for i, paper in enumerate(papers, start=1):
        print(f"\n[{i}] {paper.title}")
        print(f"    {', '.join(paper.authors[:3])}{' et al.' if len(paper.authors) > 3 else ''} ({paper.year})")
        print(f"    {paper.abs_url}")
        print(f"    abstract: {paper.abstract[:150]}...")


if __name__ == "__main__":
    main()
