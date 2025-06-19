import asyncio
import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore, TaskUpdater
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
    MessageSendParams,
    Part,
    TaskState,
    TextPart,
)
from a2a.utils import new_agent_text_message, new_task
from google.adk.artifacts import InMemoryArtifactService
from google.adk.memory.in_memory_memory_service import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

# Ensure ADKAgentExecutor uses __init__ if that was corrected in the notebook
class ADKAgentExecutor(AgentExecutor):
    def __init__(self, agent, status_message="Processing request...", artifact_name="response"):
        """Initialize a generic ADK agent executor.

        Args:
            agent: The ADK agent instance
            status_message: Message to display while processing
            artifact_name: Name for the response artifact
        """
        self.agent = agent
        self.status_message = status_message
        self.artifact_name = artifact_name
        self.runner = Runner(
            app_name=agent.name,
            agent=agent,
            artifact_service=InMemoryArtifactService(),
            session_service=InMemorySessionService(),
            memory_service=InMemoryMemoryService(),
        )

    async def cancel(self, task_id: str) -> None:
        """Cancel the execution of a specific task."""
        # Implementation for cancelling tasks

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        task = context.current_task or new_task(context.message)
        await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.contextId)

        try:
            # Update status with custom message
            await updater.update_status(
                TaskState.working,
                new_agent_text_message(self.status_message, task.contextId, task.id)
            )

            # Process with ADK agent
            session = await self.runner.session_service.create_session(
                app_name=self.agent.name,
                user_id="a2a_user",
                state={},
                session_id=task.contextId,
            )

            content = types.Content(
                role='user',
                parts=[types.Part.from_text(text=query)]
            )

            response_text = ""
            async for event in self.runner.run_async(
                user_id="a2a_user",
                session_id=session.id,
                new_message=content
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    for part in event.content.parts:
                        if hasattr(part, 'text') and part.text:
                            response_text += part.text + '\\n'
                        elif hasattr(part, 'function_call'):
                            # Log or handle function calls if needed
                            pass # Function calls are handled internally by ADK

            # Add response as artifact with custom name
            await updater.add_artifact(
                [Part(root=TextPart(text=response_text))],
                name=self.artifact_name
            )

            await updater.complete()

        except Exception as e:
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(f"Error: {e!s}", task.contextId, task.id),
                final=True
            )

def create_agent_a2a_server(
    agent,
    name,
    description,
    skills,
    host="localhost",
    port=10020,
    status_message="Processing request...",
    artifact_name="response"
):
    """Create an A2A server for any ADK agent.

    Args:
        agent: The ADK agent instance
        name: Display name for the agent
        description: Agent description
        skills: List of AgentSkill objects
        host: Server host
        port: Server port
        status_message: Message shown while processing
        artifact_name: Name for response artifacts

    Returns:
        A2AStarletteApplication instance
    """
    # Agent capabilities
    capabilities = AgentCapabilities(streaming=True)

    # Agent card (metadata)
    agent_card = AgentCard(
        name=name,
        description=description,
        url=f"http://{host}:{port}/",
        version="1.0.0",
        defaultInputModes=["text", "text/plain"],
        defaultOutputModes=["text", "text/plain"],
        capabilities=capabilities,
        skills=skills,
    )

    # Create executor with custom parameters
    executor = ADKAgentExecutor(
        agent=agent,
        status_message=status_message,
        artifact_name=artifact_name
    )

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    # Create A2A application
    return A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
