import logging
import os
import sys
import uvicorn
import httpx

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryPushNotifier, InMemoryTaskStore
from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from dotenv import load_dotenv

from langgraph_agent_executor import LangGraphDataProcessorExecutor

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MissingAPIKeyError(Exception):
    """Exception for missing API key."""


def create_langgraph_data_processor_server(host="localhost", port=10032):
    """Create A2A server for LangGraph Data Processing Agent."""
    try:
        # Check for required environment variables
        model_source = os.getenv('model_source', 'google')
        if model_source == 'google':
            if not os.getenv('GOOGLE_API_KEY'):
                raise MissingAPIKeyError(
                    'GOOGLE_API_KEY environment variable not set.'
                )
        else:
            if not os.getenv('TOOL_LLM_URL'):
                raise MissingAPIKeyError(
                    'TOOL_LLM_URL environment variable not set.'
                )
            if not os.getenv('TOOL_LLM_NAME'):
                raise MissingAPIKeyError(
                    'TOOL_LLM_NAME environment variable not set.'
                )

        # Agent capabilities
        capabilities = AgentCapabilities(streaming=True, pushNotifications=True)
        
        # Agent skills
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

        # Agent card (metadata)
        agent_card = AgentCard(
            name='LangGraph Data Processor',
            description='Advanced data processing agent using LangGraph for sophisticated analysis workflows including entity extraction, sentiment analysis, summarization, and categorization',
            url=f'http://{host}:{port}/',
            version='1.0.0',
            defaultInputModes=['text', 'text/plain'],
            defaultOutputModes=['text', 'text/plain'],
            capabilities=capabilities,
            skills=skills,
        )

        # Create executor
        executor = LangGraphDataProcessorExecutor()

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

        return server

    except MissingAPIKeyError as e:
        logger.error(f'Error: {e}')
        raise
    except Exception as e:
        logger.error(f'An error occurred during server setup: {e}')
        raise


def main():
    """Start the LangGraph Data Processor server."""
    host = os.getenv('HOST', 'localhost')
    port = int(os.getenv('PORT', '10032'))
    
    try:
        server = create_langgraph_data_processor_server(host, port)
        logger.info(f"Starting LangGraph Data Processor server on {host}:{port}")
        uvicorn.run(server.build(), host=host, port=port)
    except Exception as e:
        logger.error(f'Failed to start server: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main() 