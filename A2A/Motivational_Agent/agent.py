import os
import random

from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm

# https://docs.litellm.ai/docs/providers/openrouter
# model = LiteLlm(
#     model="qwen-qwq-32b",
#     provider="groq",
#     api_key=os.getenv("GROQ_API_KEY"),
# )


def get_motivational_quote():
    quotes = [
        "The best way to get started is to quit talking and begin doing. - Walt Disney",
        "The pessimist sees difficulty in every opportunity. The optimist sees opportunity in every difficulty. - Winston Churchill",
        "Don’t let yesterday take up too much of today. - Will Rogers",
        "You learn more from failure than from success. Don’t let it stop you. Failure builds character. - Unknown",
    ]
    return random.choice(quotes)


root_agent = Agent(
    name="Motivational_Agent",
    model='gemini-2.0-flash-001',
    description="Motivational quote agent",
    instruction="""
    You are a helpful assistant that provides motivational quotes. 
    """,
    tools=[get_motivational_quote],
)