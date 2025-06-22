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
    data_processing_agent,
    report_writer_agent
)
from server_utils import create_agent_a2a_server # Assuming AgentSkill is also in server_utils or imported there
from a2a.types import AgentSkill # Explicitly import AgentSkill if not re-exported by server_utils

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
    port = AGENT_PORTS["DataProcessingAgent"]
    return create_agent_a2a_server(
        agent=data_processing_agent,
        name="Data Processing Agent",
        description="Processes and analyzes data, extracts insights.",
        skills=[
            AgentSkill(
                id="process_data",
                name="Process Data",
                description="Analyzes provided data to extract key information, trends, or summaries.",
                tags=["data analysis", "text processing", "insight extraction"],
                examples=["Extract company names from the provided text.", "Summarize these findings: [data]"]
            )
        ],
        host=HOST,
        port=port,
        status_message="Data Processor is analyzing...",
        artifact_name="processed_data_insights"
    )

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

    # Start all agent servers
    pm_thread = run_agent_server_in_background(create_pm_agent_server, AGENT_PORTS["ProjectManagerAgent"], "ProjectManagerAgent")
    ra_thread = run_agent_server_in_background(create_ra_agent_server, AGENT_PORTS["ResearchAnalystAgent"], "ResearchAnalystAgent")
    dp_thread = run_agent_server_in_background(create_dp_agent_server, AGENT_PORTS["DataProcessingAgent"], "DataProcessingAgent")
    rw_thread = run_agent_server_in_background(create_rw_agent_server, AGENT_PORTS["ReportWriterAgent"], "ReportWriterAgent")

    print("\nWaiting for servers to initialize (e.g., 5-10 seconds)...")
    time.sleep(10) # Give servers time to start up

    print("\nChecking if server threads are alive:")
    for agent_name, thread in [("PM", pm_thread), ("RA", ra_thread), ("DP", dp_thread), ("RW", rw_thread)]:
        if thread.is_alive():
            print(f"- {agent_name} server thread is alive.")
        else:
            print(f"- {agent_name} server thread is NOT alive. Check logs.")

    print("\nRegistering specialized agents with Project Manager's A2AToolClient...")
    # The ProjectManagerAgent (via its A2A server) will use pm_a2a_client internally.
    # For the PM to *discover* other agents, its pm_a2a_client needs to know their URLs.
    pm_a2a_client.add_remote_agent(f"http://{HOST}:{AGENT_PORTS['ResearchAnalystAgent']}")
    pm_a2a_client.add_remote_agent(f"http://{HOST}:{AGENT_PORTS['DataProcessingAgent']}")
    pm_a2a_client.add_remote_agent(f"http://{HOST}:{AGENT_PORTS['ReportWriterAgent']}")

    print("Remote agents registered with PM's client:")
    # This list_remote_agents call is directly on the object, not via A2A call to PM yet.
    # This is to confirm client-side registration.
    # In a real scenario, the PM agent would call its own tool version of this.

    # The list_remote_agents in the prompt expects a dictionary, but my A2AToolClient.list_remote_agents
    # was modified to return a list of agent_data. I'll adjust the print statement here.
    registered_agents_on_pm_client = pm_a2a_client.list_remote_agents()
    print(f"PM's A2AToolClient knows about: {[(agent.get('name', 'Unknown Name'), agent.get('url', 'Unknown URL')) for agent in registered_agents_on_pm_client]}")


    # Placeholder for actual testing (Step 7 of the main plan)
    print("\n--- Main Orchestrator Setup Complete ---")
    print("To test, you would now send a request to the ProjectManagerAgent's A2A endpoint, e.g.:")
    print(f"http://{HOST}:{AGENT_PORTS['ProjectManagerAgent']}/")
    print("Example test query for PM: 'Generate a report on the benefits of AI in customer service.'")

    # Keep main thread alive to let background servers run
    # In a real deployment, the servers would run indefinitely until stopped.
    # For this script, we might want it to exit after some time or a manual stop.
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down main orchestrator and attempting to stop server threads...")
    finally:
        # Uvicorn servers started with daemon=True threads should exit when the main program exits.
        # However, explicit cleanup might be needed in more complex scenarios.
        print("Main orchestrator stopped.")
