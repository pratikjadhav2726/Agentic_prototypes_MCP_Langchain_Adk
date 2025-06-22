import asyncio
import threading
import time
import nest_asyncio
import uvicorn # uvicorn needs to be available
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up Google API configuration for Gemini
os.environ['GOOGLE_GENAI_USE_VERTEXAI'] = 'FALSE'  # Use direct Gemini API
api_key = os.getenv('GOOGLE_API_KEY')  # Your Gemini API key
if api_key:
    os.environ['GOOGLE_API_KEY'] = api_key

# Imports from the new package
from a2a_client import A2AToolClient
from agents import (
    project_manager_agent,
    research_analyst_agent,
    report_writer_agent
)
from server_utils import create_agent_a2a_server # Assuming AgentSkill is also in server_utils or imported there
from a2a.types import AgentSkill # Explicitly import AgentSkill if not re-exported by server_utils
from langgraph_server_utils import create_data_processor_server

# Apply nest_asyncio for environments like Jupyter, but also generally good for script-based asyncio with uvicorn threads
nest_asyncio.apply()

# --- Global A2A Client for Project Manager ---
# The ProjectManagerAgent will use this client to talk to other agents.
# Its tools (list_remote_agents, create_task) are methods of this client instance.
pm_a2a_client = A2AToolClient()

# --- Instantiate ProjectManagerAgent with its tools ---
# The actual Agent object needs to be configured with its tools.
# The 'tools' parameter in ADK's Agent class expects a list of callable functions or methods.
project_manager_agent.tools = [pm_a2a_client.list_remote_agents, pm_a2a_client.create_task]
print(f"ProjectManagerAgent instantiated with tools: {project_manager_agent.tools}")


# --- A2A Server Creation Functions for each agent ---

# Configuration for agent ports
AGENT_PORTS = {
    "ProjectManagerAgent": 10030,
    "ResearchAnalystAgent": 10031,
    "DataProcessingAgent": 10032,
    "ReportWriterAgent": 10033,
}
HOST = "127.0.0.1" # Or "localhost"

def create_pm_agent_server():
    port = AGENT_PORTS["ProjectManagerAgent"]
    return create_agent_a2a_server(
        agent=project_manager_agent,
        name="Project Manager Agent",
        description="Orchestrates research, data processing, and report writing tasks.",
        skills=[
            AgentSkill(
                id="manage_project",
                name="Manage Project",
                description="Manages a multi-step project involving research, data analysis, and report generation.",
                tags=["orchestration", "project management", "coordination"],
                examples=[
                    "Generate a report on the impact of AI on healthcare.",
                    "Analyze market trends for renewable energy and produce a summary."
                ]
            )
        ],
        host=HOST,
        port=port,
        status_message="Project Manager is coordinating...",
        artifact_name="final_project_report"
    )

def create_ra_agent_server():
    port = AGENT_PORTS["ResearchAnalystAgent"]
    return create_agent_a2a_server(
        agent=research_analyst_agent,
        name="Research Analyst Agent",
        description="Gathers information from the web on specified topics.",
        skills=[
            AgentSkill(
                id="conduct_research",
                name="Conduct Research",
                description="Searches the web for articles, studies, and data on a given subject.",
                tags=["research", "web search", "information gathering"],
                examples=["Find information on quantum computing breakthroughs in 2023."]
            )
        ],
        host=HOST,
        port=port,
        status_message="Research Analyst is searching...",
        artifact_name="research_findings"
    )

def create_dp_agent_server():
    """Create LangGraph-based Data Processing Agent server using langgraph_server_utils."""
    port = AGENT_PORTS["DataProcessingAgent"]
    return create_data_processor_server(HOST, port)

def create_rw_agent_server():
    port = AGENT_PORTS["ReportWriterAgent"]
    return create_agent_a2a_server(
        agent=report_writer_agent,
        name="Report Writer Agent",
        description="Compiles information into structured reports.",
        skills=[
            AgentSkill(
                id="write_report",
                name="Write Report",
                description="Generates a formatted textual report from provided content and structure.",
                tags=["reporting", "text generation", "compilation"],
                examples=["Compile a report with sections: Intro, Findings, Conclusion. Content: [data]"]
            )
        ],
        host=HOST,
        port=port,
        status_message="Report Writer is drafting...",
        artifact_name="compiled_report"
    )

# --- Helper to run servers in background threads (similar to notebook) ---
# Store server threads globally to manage them if needed, though uvicorn might handle shutdown with daemon threads.
server_threads = []

async def run_server_async(create_server_func, port, server_name):
    print(f"Attempting to start {server_name} on port {port}...")
    app = create_server_func() # This should return A2AStarletteApplication

    # Check if app has .build() method, if it's already a Starlette app, it might not.
    built_app = app.build() if hasattr(app, 'build') else app

    config = uvicorn.Config(
        built_app,
        host=HOST,
        port=port,
        log_level="info", # Use 'info' for more startup details, 'error' for less
        loop="asyncio"
    )
    server = uvicorn.Server(config)

    try:
        await server.serve()
    except Exception as e:
        print(f"Error starting {server_name} on port {port}: {e}")
    finally:
        print(f"{server_name} on port {port} has shut down.")


def run_agent_server_in_background(create_server_func, port, name):
    def run_in_new_loop():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(run_server_async(create_server_func, port, name))
        finally:
            loop.close()

    thread = threading.Thread(target=run_in_new_loop, daemon=True, name=f"{name}Thread")
    thread.start()
    server_threads.append(thread)
    print(f"{name} server thread started.")
    return thread

# --- Main Execution Block ---
if __name__ == "__main__":
    print("Starting AI Company Agent Servers...")
    print("Note: Data Processing Agent now uses LangGraph with langgraph_server_utils!")

    # Start all agent servers
    pm_thread = run_agent_server_in_background(create_pm_agent_server, AGENT_PORTS["ProjectManagerAgent"], "ProjectManagerAgent")
    ra_thread = run_agent_server_in_background(create_ra_agent_server, AGENT_PORTS["ResearchAnalystAgent"], "ResearchAnalystAgent")
    dp_thread = run_agent_server_in_background(create_dp_agent_server, AGENT_PORTS["DataProcessingAgent"], "LangGraphDataProcessorAgent")
    rw_thread = run_agent_server_in_background(create_rw_agent_server, AGENT_PORTS["ReportWriterAgent"], "ReportWriterAgent")

    print("\nWaiting for servers to initialize (e.g., 5-10 seconds)...")
    time.sleep(10) # Give servers time to start up

    print("\nChecking if server threads are alive:")
    for agent_name, thread in [("PM", pm_thread), ("RA", ra_thread), ("DP", dp_thread), ("RW", rw_thread)]:
        if thread.is_alive():
            print(f"✅ {agent_name} server thread is running")
        else:
            print(f"❌ {agent_name} server thread is not running")

    print("\nAll servers should now be running!")
    print("You can test them using the test script or the chatbot UI.")
    print("\nPress Ctrl+C to stop all servers.")

    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        # The daemon threads will be terminated when the main thread exits