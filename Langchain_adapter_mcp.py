
# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
import os
import asyncio
from dotenv import load_dotenv
import json
load_dotenv()
from langchain_groq import ChatGroq
model = ChatGroq(
            temperature=0, 
            groq_api_key=os.getenv("GROQ_API_KEY"), 
            model_name="qwen-qwq-32b"
        )
# Load MCP server configuration from JSON file
def load_mcp_server_config(config_path="mcp_server_config.json"):
    try:
        with open(config_path, "r") as config_file:
            config = json.load(config_file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_path}' not found.")
    except json.JSONDecodeError as e:
        raise ValueError(f"Error decoding JSON from '{config_path}': {e}")


async def main():
    config = load_mcp_server_config(os.getenv("MCP_Config"))
    async with MultiServerMCPClient(
      config.get("mcpServers")
    )as client:
        agent = create_react_agent(model, client.get_tools())
        math_response = await agent.ainvoke({"messages": "what's (3 + 5) x 12?"})
        print(f"Agent response: {math_response}")
        Browser_automation = await agent.ainvoke({"messages": "go to https://www.google.com'."})
        print(f"Agent response: {Browser_automation}")
# Run the async main function
asyncio.run(main())