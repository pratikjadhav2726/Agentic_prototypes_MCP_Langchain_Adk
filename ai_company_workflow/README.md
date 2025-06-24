# AI Company Workflow

This workflow demonstrates a sample implementation of an AI agents company using the MCP and A2A. It consists of several agents that collaborate to perform tasks such as research, data processing, and report writing.

## Components

*   **Project Manager Agent**: Orchestrates the workflow and delegates tasks to other agents.
*   **Research Analyst Agent**: Gathers information from the web.
*   **Data Processing Agent**: Processes and analyzes data.
*   **Report Writer Agent**: Compiles information into structured reports.

## How to Run

1.  Ensure you have Python installed and the necessary dependencies (see `requirements.txt`).
2.  Navigate to the `ai_company_workflow` directory.
3.  Run the `main_orchestrator.py` script:

    ```bash
    python main_orchestrator.py
    ```

    This will start all the agent servers.
4.  You can then send requests to the Project Manager Agent's A2A endpoint (e.g., `http://127.0.0.1:10030/`) to initiate a workflow. For example:

    "Generate a report on the benefits of AI in customer service."

## A2A Client

The `a2a_client.py` provides a client utility to interact with A2A-enabled agents. It can be used to:
- Add and list remote agents.
- Create tasks (send messages) to agents.
- Remove remote agents.

## Server Utilities

The `server_utils.py` contains helper functions to create A2A servers for ADK (Agent Development Kit) agents. It simplifies the process of exposing an ADK agent via the A2A protocol.
