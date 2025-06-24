import json
import uuid
from typing import Any
import httpx
import requests
from a2a.client import A2AClient
from a2a.types import AgentCard, SendMessageRequest, MessageSendParams # Make sure AgentCard is imported if used directly

# Ensure A2AToolClient uses __init__ if that was corrected in the notebook
class A2AToolClient:
    """A2A client."""

    def __init__(self, default_timeout: float = 120.0):
        # Cache for agent metadata - also serves as the list of registered agents
        # None value indicates agent is registered but metadata not yet fetched
        self._agent_info_cache: dict[str, dict[str, Any] | None] = {}
        # Default timeout for requests (in seconds)
        self.default_timeout = default_timeout

    def add_remote_agent(self, agent_url: str):
        """Add agent to the list of available remote agents."""
        normalized_url = agent_url.rstrip('/')
        if normalized_url not in self._agent_info_cache:
            # Initialize with None to indicate metadata not yet fetched
            self._agent_info_cache[normalized_url] = None

    def list_remote_agents(self) -> list[dict[str, Any]]:
        """List available remote agents with caching."""
        if not self._agent_info_cache:
            return [] # Return empty list if no agents are registered

        # Create a list to store agent data, fetching if not cached
        # This part was modified to correctly return a list of agent cards / info
        # and handle the cache logic as intended in the notebook.
        agent_infos = []
        for url, cached_info in self. _agent_info_cache.items():
            if cached_info is not None:
                agent_infos.append(cached_info)
            else:
                try:
                    response = requests.get(f"{url}/.well-known/agent.json", timeout=5) # Added timeout
                    response.raise_for_status() # Raise an exception for bad status codes
                    agent_data = response.json()
                    self._agent_info_cache[url] = agent_data # Cache the fetched data
                    agent_infos.append(agent_data)
                except requests.RequestException as e: # Catch network/request errors
                    print(f"Failed to fetch agent info from {url}: {e}")
                except json.JSONDecodeError as e: # Catch errors in parsing JSON
                    print(f"Failed to parse agent info from {url}: {e}")
        return agent_infos


    async def create_task(self, agent_url: str, message: str) -> str:
        """Send a message following the official A2A SDK pattern."""
        # Configure httpx client with timeout
        timeout_config = httpx.Timeout(
            timeout=self.default_timeout,
            connect=10.0,
            read=self.default_timeout,
            write=10.0,
            pool=5.0
        )

        async with httpx.AsyncClient(timeout=timeout_config) as httpx_client:
            # Check if we have cached agent card data
            normalized_url = agent_url.rstrip('/')
            if normalized_url in self._agent_info_cache and self._agent_info_cache[normalized_url] is not None:
                agent_card_data = self._agent_info_cache[normalized_url]
            else:
                # Fetch the agent card
                try:
                    agent_card_response = await httpx_client.get(f"{normalized_url}/.well-known/agent.json")
                    agent_card_response.raise_for_status()
                    agent_card_data = agent_card_response.json()
                    self._agent_info_cache[normalized_url] = agent_card_data # Cache it
                except httpx.RequestError as e:
                    return f"Error fetching agent card from {normalized_url}: {e}"
                except json.JSONDecodeError as e:
                    return f"Error parsing agent card JSON from {normalized_url}: {e}"


            # Create AgentCard from data
            try:
                agent_card = AgentCard(**agent_card_data)
            except Exception as e: # Catch error if agent_card_data is not valid for AgentCard
                return f"Error creating AgentCard from fetched data for {normalized_url}: {e}. Data: {agent_card_data}"

            # Create A2A client with the agent card
            client = A2AClient(
                httpx_client=httpx_client,
                agent_card=agent_card
            )

            # Build the message parameters following official structure
            send_message_payload = {
                'message': {
                    'role': 'user',
                    'parts': [
                        {'kind': 'text', 'text': message}
                    ],
                    'messageId': uuid.uuid4().hex,
                }
            }

            # Create the request
            request = SendMessageRequest(
                id=str(uuid.uuid4()),
                params=MessageSendParams(**send_message_payload)
            )

            # Send the message with timeout configuration
            try:
                response = await client.send_message(request)
            except httpx.RequestError as e:
                return f"Error sending message to {normalized_url}: {e}"


            # Extract text from response
            try:
                response_dict = response.model_dump(mode='json', exclude_none=True)
                if 'result' in response_dict and 'artifacts' in response_dict['result']:
                    artifacts = response_dict['result']['artifacts']
                    for artifact in artifacts:
                        if 'parts' in artifact:
                            for part_item in artifact['parts']:
                                if 'text' in part_item:
                                    return part_item['text']

                # If we couldn't extract text, return the full response as formatted JSON
                return json.dumps(response_dict, indent=2)

            except Exception as e:
                # Log the error and return string representation
                print(f"Error parsing response: {e}")
                return str(response)

    def remove_remote_agent(self, agent_url: str):
        """Remove an agent from the list of available remote agents."""
        normalized_url = agent_url.rstrip('/')
        if normalized_url in self._agent_info_cache:
            del self._agent_info_cache[normalized_url]
