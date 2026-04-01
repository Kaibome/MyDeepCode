import unittest

from deepcode.agent_flow import AgentAggregation
from deepcode.agents import AgentConfig, ReActAgent


class AgentAggregationTest(unittest.IsolatedAsyncioTestCase):
    @unittest.skip("require llm config and runnable mcp server")
    async def test_agent_aggregation(self):
        base_cfg = {
            "model_provider": "openai",
            "api_key": "please provide your api_key",
            "api_base": "please provide your api_base",
            "model_name": "please provide your model_name",
        }
        agent_1 = ReActAgent(
            AgentConfig(
                name="agent_1",
                llm_config=base_cfg,
                system_prompt="You are an assistant that can use tools.",
            )
        )
        agent_2 = ReActAgent(
            AgentConfig(
                name="agent_2",
                llm_config=base_cfg,
                system_prompt="You are an assistant that can use tools.",
            )
        )
        aggregator = ReActAgent(
            AgentConfig(
                name="agent_3",
                llm_config=base_cfg,
                system_prompt="You summarize outputs from other agents.",
            )
        )

        await agent_1.add_mcp_servers(
            [
                {
                    "server_name": "command_executor",
                    "command": "python",
                    "args": ["provide your mcp server path"],
                }
            ]
        )

        aggregation = AgentAggregation(aggregator=aggregator, source_agents=[agent_1, agent_2])
        result = await aggregation.ainvoke(
            {"query": "Check current directory and list files."}
        )
        print(result)
