import unittest

from deepcode.agents import AgentConfig, ReActAgent


class ReactAgentTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.agent = ReActAgent(
            AgentConfig(
                name="default_agent",
                llm_config={
                    "model_provider": "openai",
                    "api_key": "please provide your api_key",
                    "api_base": "please provide your api_base",
                    "model_name": "please provide your model_name",
                },
            )
        )

    @unittest.skip("require llm config")
    def test_call_model(self):
        response = self.agent.call_llm(
            model_name="gpt-4o-mini",
            messages=[{"role": "user", "content": "hello"}],
        )
        self.assertTrue(bool(response.content))

    @unittest.skip("require stdio mcp server path")
    async def test_mcp_tool(self):
        await self.agent.add_mcp_servers(
            [
                {
                    "server_name": "command_executor",
                    "command": "python",
                    "args": ["provide your mcp server path"],
                }
            ]
        )
        result = await self.agent.execute_mcp_tool("replace with tool name", {})
        print(result)

    @unittest.skip("require llm config and stdio mcp server path")
    async def test_react_loop(self):
        await self.agent.add_mcp_servers(
            [
                {
                    "server_name": "command_executor",
                    "command": "python",
                    "args": ["provide your mcp server path"],
                }
            ]
        )
        result = await self.agent.ainvoke(
            {"query": "What is the current working directory and what files are in it?"}
        )
        print(result)
