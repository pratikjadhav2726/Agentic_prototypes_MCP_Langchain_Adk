import asyncio
from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioServerParameters
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.artifacts import InMemoryArtifactService
from google.genai.types import Content, Part

# Initialize the model
model = LiteLlm(model="groq/qwen-qwq-32b")

# Asynchronous function to get tools from the MCP server
async def get_tools_async():
    tools, exit_stack = await MCPToolset.from_server(
        connection_params=StdioServerParameters(
            command='python',
            args=["/Users/pratik/Documents/MCP_Agent/A2A/Motivational_Agent/motivational_quotes_mcp_server.py"]
        )
    )
    return tools, exit_stack

# Asynchronous function to create the agent
async def create_agent():
    tools, exit_stack = await get_tools_async()
    agent = Agent(
        name="Motivational_Agent",
        model=model,
        description="Motivational quote agent",
        instruction="You are a helpful assistant that provides motivational quotes.",
        tools=tools,
    )
    return agent, exit_stack

# Asynchronous main function to run the agent
async def async_main():
    # Initialize services
    session_service = InMemorySessionService()
    artifact_service = InMemoryArtifactService()

    # Create a session
    session = session_service.create_session(
        state={},
        app_name='motivational_agent',
        user_id='user_1',
        session_id='session_1',
    )

    # Define the user query
    query = "Can you provide inspirational and motivational quote to start my day?"
    content = Content(role='user', parts=[Part(text=query)])

    # Create the agent and get the exit stack
    agent, exit_stack = await create_agent()

    # Initialize the runner
    runner = Runner(
        app_name='motivational_agent',
        agent=agent,
        artifact_service=artifact_service,
        session_service=session_service,
    )

    # Run the agent and process the response
    events_async = runner.run_async(
        session_id=session.id,
        user_id=session.user_id,
        new_message=content
    )

    async for event in events_async:
        print(f"Event: {event.actions}")
        if event.is_final_response():
            if event.content and event.content.parts:
                final_response_text = event.content.parts[0].text
            elif event.actions and event.actions.escalate:
                final_response_text = f"Agent escalated: {event.error_message or 'No specific message.'}"
            print(f"############# Final Response #############\n\n{final_response_text}")
            break

    # Close the MCP server connectionsss
    await exit_stack.aclose()

# Entry point to run the asynchronous main function
if __name__ == "__main__":
    asyncio.run(async_main())