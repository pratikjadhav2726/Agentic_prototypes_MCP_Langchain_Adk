import logging
import os
import httpx
from typing import Type, List, Optional

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv

from generic_langgraph_executor import create_langgraph_executor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


def create_langgraph_agent_a2a_server(
    agent_class: Type,
    name: str,
    description: str,
    skills: List[AgentSkill],
    host: str = "localhost",
    port: int = 10020,
    agent_name: Optional[str] = None,
    capabilities: Optional[AgentCapabilities] = None,
    input_modes: Optional[List[str]] = None,
    output_modes: Optional[List[str]] = None,
    version: str = "1.0.0",
    check_api_key: bool = True,
    api_key_env_var: str = "GOOGLE_API_KEY"
):
    """Create an A2A server for any LangGraph agent.

    This function provides a unified way to create A2A servers for LangGraph agents,
    similar to how server_utils.py works for ADK agents.

    Args:
        agent_class: The LangGraph agent class to instantiate
        name: Display name for the agent
        description: Agent description
        skills: List of AgentSkill objects
        host: Server host (default: "localhost")
        port: Server port (default: 10020)
        agent_name: Optional name for the agent (defaults to class name)
        capabilities: Optional AgentCapabilities (defaults to streaming=True, pushNotifications=True)
        input_modes: Optional list of input modes (defaults to ["text", "text/plain"])
        output_modes: Optional list of output modes (defaults to ["text", "text/plain"])
        version: Agent version (default: "1.0.0")
        check_api_key: Whether to check for API key (default: True)
        api_key_env_var: Environment variable name for API key (default: "GOOGLE_API_KEY")

    Returns:
        A2AStarletteApplication instance

    Raises:
        MissingAPIKeyError: If API key is required but not found
        Exception: If server setup fails
    """
    try:
        # Check for required environment variables if needed
        if check_api_key and not os.getenv(api_key_env_var):
            raise MissingAPIKeyError(
                f'{api_key_env_var} environment variable not set.'
            )

        # Set default values
        if capabilities is None:
            capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        
        if input_modes is None:
            input_modes = ["text", "text/plain"]
            
        if output_modes is None:
            output_modes = ["text", "text/plain"]
            
        if agent_name is None:
            agent_name = agent_class.__name__

        # Agent card (metadata)
        agent_card = AgentCard(
            name=name,
            description=description,
            url=f'http://{host}:{port}/',
            version=version,
            defaultInputModes=input_modes,
            defaultOutputModes=output_modes,
            capabilities=capabilities,
            skills=skills,
        )

        # Create executor using the generic executor
        executor = create_langgraph_executor(agent_class, agent_name)

        # Create request handler
        httpx_client = httpx.AsyncClient()
        request_handler = DefaultRequestHandler(
            agent_executor=executor,
            task_store=InMemoryTaskStore(),
            push_notifier=InMemoryPushNotifier(httpx_client),
        )

        # Create A2A application
        server = A2AStarletteApplication(
            agent_card=agent_card,
            http_handler=request_handler
        )

        logger.info(f"Created A2A server for {name} on {host}:{port}")
        return server

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        raise
    except Exception as e:
        logger.error(f'An error occurred during server setup: {e}')
        raise


def create_data_processor_server(
    host: str = "localhost",
    port: int = 10032,
    agent_class=None
):
    """Create A2A server for Data Processing Agent using LangGraph.
    
    Args:
        host: Server host
        port: Server port
        agent_class: Optional custom agent class (defaults to LangGraphDataProcessor)
        
    Returns:
        A2AStarletteApplication instance
    """
    from langgraph_data_processor import LangGraphDataProcessor
    
    if agent_class is None:
        agent_class = LangGraphDataProcessor
    
    skills = [
        AgentSkill(
            id='advanced_data_analysis',
            name='Advanced Data Analysis',
            description='Performs comprehensive data analysis including entity extraction, sentiment analysis, summarization, and content categorization',
            tags=['data analysis', 'entity extraction', 'sentiment analysis', 'summarization', 'categorization'],
            examples=[
                'Analyze this text for entities and sentiment',
                'Extract key insights from this document',
                'Categorize and summarize this content'
            ],
        ),
        AgentSkill(
            id='entity_extraction',
            name='Entity Extraction',
            description='Extracts named entities like companies, people, locations, dates, and numbers from text',
            tags=['entity extraction', 'named entities', 'text analysis'],
            examples=[
                'Extract all company names from this text',
                'Find all people mentioned in this document',
                'Identify dates and locations in this content'
            ],
        ),
        AgentSkill(
            id='sentiment_analysis',
            name='Sentiment Analysis',
            description='Analyzes the sentiment and emotional tone of text content',
            tags=['sentiment analysis', 'emotion detection', 'text analysis'],
            examples=[
                'What is the sentiment of this text?',
                'Analyze the emotional tone of this content',
                'Is this text positive, negative, or neutral?'
            ],
        ),
        AgentSkill(
            id='content_summarization',
            name='Content Summarization',
            description='Creates concise summaries of longer text content',
            tags=['summarization', 'text compression', 'key points'],
            examples=[
                'Summarize this long document',
                'Create a brief summary of this text',
                'Extract the key points from this content'
            ],
        ),
        AgentSkill(
            id='content_categorization',
            name='Content Categorization',
            description='Categorizes content into different topics and domains',
            tags=['categorization', 'topic classification', 'content analysis'],
            examples=[
                'What category does this content belong to?',
                'Classify this text by topic',
                'Determine the domain of this content'
            ],
        )
    ]

    return create_langgraph_agent_a2a_server(
        agent_class=agent_class,
        name="LangGraph Data Processor",
        description="Advanced data processing agent using LangGraph for sophisticated analysis workflows including entity extraction, sentiment analysis, summarization, and categorization",
        skills=skills,
        host=host,
        port=port,
        agent_name="LangGraph Data Processor"
    )


def create_currency_agent_server(
    host: str = "localhost",
    port: int = 10034,
    agent_class=None
):
    """Create A2A server for Currency Agent using LangGraph.
    
    Args:
        host: Server host
        port: Server port
        agent_class: Optional custom agent class (defaults to CurrencyAgent)
        
    Returns:
        A2AStarletteApplication instance
    """
    from example_currency_agent import CurrencyAgent
    
    if agent_class is None:
        agent_class = CurrencyAgent
    
    skills = [
        AgentSkill(
            id='currency_conversion',
            name='Currency Conversion',
            description='Converts between different currencies using real-time exchange rates',
            tags=['currency', 'conversion', 'exchange rates', 'finance'],
            examples=[
                'Convert 100 USD to EUR',
                'What is the exchange rate between GBP and JPY?',
                'How much is 50 CAD in USD?'
            ],
        ),
        AgentSkill(
            id='exchange_rate_lookup',
            name='Exchange Rate Lookup',
            description='Looks up current exchange rates for currency pairs',
            tags=['exchange rates', 'currency pairs', 'financial data'],
            examples=[
                'Get the current USD to EUR exchange rate',
                'What is the GBP to USD rate today?',
                'Check the exchange rate for EUR to JPY'
            ],
        )
    ]

    return create_langgraph_agent_a2a_server(
        agent_class=agent_class,
        name="LangGraph Currency Agent",
        description="Specialized currency conversion agent using LangGraph for real-time exchange rate lookups and currency conversions",
        skills=skills,
        host=host,
        port=port,
        agent_name="LangGraph Currency Agent"
    )


def create_research_agent_server(
    host: str = "localhost",
    port: int = 10035,
    agent_class=None
):
    """Create A2A server for Research Agent using LangGraph.
    
    Args:
        host: Server host
        port: Server port
        agent_class: Optional custom agent class
        
    Returns:
        A2AStarletteApplication instance
    """
    # This would be implemented when you create a ResearchAgent class
    # For now, this is a placeholder showing the pattern
    
    skills = [
        AgentSkill(
            id='web_research',
            name='Web Research',
            description='Conducts comprehensive web research on specified topics',
            tags=['research', 'web search', 'information gathering', 'analysis'],
            examples=[
                'Research the latest AI developments',
                'Find information about renewable energy trends',
                'Gather data on market trends in technology'
            ],
        ),
        AgentSkill(
            id='source_validation',
            name='Source Validation',
            description='Validates and evaluates the credibility of information sources',
            tags=['validation', 'credibility', 'fact-checking', 'sources'],
            examples=[
                'Validate the credibility of this source',
                'Check if this information is from a reliable source',
                'Evaluate the trustworthiness of this data'
            ],
        )
    ]

    # Placeholder - would need actual ResearchAgent class
    # return create_langgraph_agent_a2a_server(
    #     agent_class=agent_class or ResearchAgent,
    #     name="LangGraph Research Agent",
    #     description="Advanced research agent using LangGraph for comprehensive web research and source validation",
    #     skills=skills,
    #     host=host,
    #     port=port,
    #     agent_name="LangGraph Research Agent"
    # )
    
    raise NotImplementedError("ResearchAgent class not yet implemented")


# Utility function to create multiple servers at once
def create_multiple_langgraph_servers(server_configs: List[dict]):
    """Create multiple LangGraph agent servers from configuration.
    
    Args:
        server_configs: List of dictionaries with server configuration
                       Each dict should have: agent_class, name, description, skills, host, port
        
    Returns:
        List of A2AStarletteApplication instances
    """
    servers = []
    
    for config in server_configs:
        server = create_langgraph_agent_a2a_server(**config)
        servers.append(server)
    
    return servers


# Example usage:
# from langgraph_data_processor import LangGraphDataProcessor
# server = create_langgraph_agent_a2a_server(
#     agent_class=LangGraphDataProcessor,
#     name="My Custom Agent",
#     description="A custom LangGraph agent",
#     skills=[...],
#     host="localhost",
#     port=10040
# )