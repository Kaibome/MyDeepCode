#!/usr/bin/env python
# coding: utf-8

import asyncio
import os
import sys
from pathlib import Path


# Add project root to Python path for direct script execution.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deepcode.agent_flow.multi_agent_research import MultiAgentResearchFlow
from deepcode.agents.react_agent import AgentConfig, ChatAgent


async def chat_agent_demo() -> None:
    llm_config = {
        "model_provider": os.getenv("LLM_MODEL_PROVIDER"),
        "api_key": os.getenv("LLM_API_KEY"),
        "api_base": os.getenv("LLM_API_BASE"),
        "model_name": os.getenv("LLM_MODEL_NAME"),
    }

    agent = ChatAgent(
        AgentConfig(
            name="test_chat_agent",
            llm_config=llm_config,
            system_prompt="You are a helpful assistant.",
        )
    )

    query = "Please explain what artificial intelligence is in one paragraph."
    result = await agent.ainvoke({"query": query})
    print("Query:", query)
    print("Answer:", result.get("content", ""))


async def pipeline_demo(input_source: str) -> None:
    flow = MultiAgentResearchFlow()
    await flow.initialize_agents()
    await flow.ainvoke(input_source)


async def main() -> None:
    # Example:
    #   python -m deepcode.tests.try_multi_agent_research file:///D:/path/to/paper.pdf
    # Or:
    #   python -m deepcode.tests.try_multi_agent_research D:/path/to/paper.pdf
    if len(sys.argv) >= 2:
        source = sys.argv[1]
        await pipeline_demo(source)
    else:
        await chat_agent_demo()


if __name__ == "__main__":
    asyncio.run(main())
