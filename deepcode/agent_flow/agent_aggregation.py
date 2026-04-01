#!/usr/bin/env python
# coding: utf-8
"""
Agent aggregation pattern: fan-out to multiple source agents, then aggregate
results through a lead agent.

Mirrors the original deepcode AgentAggregation but uses the LangChain-based
BaseAgent from deepcode.agents.
"""

from deepcode.agents import BaseAgent


class AgentAggregation:
    """
    Fan-out / fan-in orchestration.

    All *source_agents* are invoked concurrently with the same inputs;
    their results are concatenated and forwarded to the *aggregator* agent.
    """

    def __init__(self, aggregator: BaseAgent, source_agents: list[BaseAgent]):
        self.aggregator = aggregator
        self.source_agents = source_agents

    async def ainvoke(self, inputs, runtime=None):
        # Launch all source agents concurrently
        results = {}
        for agent in self.source_agents:
            results[agent.name] = agent.ainvoke(inputs, runtime)

        # Await results and format
        aggregated_results = []
        for agent in self.source_agents:
            result = await results[agent.name]
            aggregated_results.append(
                f"# {agent.name}\n{str(result)}\n"
            )
        aggregated_results = "\n".join(aggregated_results)

        return await self.aggregator.ainvoke(
            {"query": aggregated_results}, runtime
        )
