#!/usr/bin/env python
# coding: utf-8

import argparse
import asyncio
import logging

from deepcode.agent_flow import MultiAgentResearchFlow


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="DeepCode LangGraph multi-agent paper-to-code pipeline"
    )
    parser.add_argument(
        "input_source",
        nargs="?",
        default="",
        help="Paper input path or URL, e.g. file:///D:/paper.pdf or https://...",
    )
    return parser


async def run(input_source: str) -> None:
    if not input_source:
        raise ValueError("input_source is required")

    flow = MultiAgentResearchFlow()
    await flow.initialize_agents()
    await flow.ainvoke(input_source)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    args = _build_parser().parse_args()
    asyncio.run(run(args.input_source))


if __name__ == "__main__":
    main()
