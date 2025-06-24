import logging
import os
import sys
import uvicorn
from dotenv import load_dotenv

from langgraph_server_utils import create_data_processor_server

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Start the LangGraph Data Processor server."""
    host = os.getenv('HOST', 'localhost')
    port = int(os.getenv('PORT', '10032'))
    
    try:
        server = create_data_processor_server(host, port)
        logger.info(f"Starting LangGraph Data Processor server on {host}:{port}")
        uvicorn.run(server.build(), host=host, port=port)
    except Exception as e:
        logger.error(f'Failed to start server: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()