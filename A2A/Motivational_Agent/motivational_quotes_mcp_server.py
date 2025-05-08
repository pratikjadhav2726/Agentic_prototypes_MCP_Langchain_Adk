from mcp.server.fastmcp import FastMCP
import random
import asyncio
import uvicorn

mcp = FastMCP("MotivationalQuotes")

@mcp.tool()
def get_motivational_quote() -> str:
    """Provide a random motivational quote"""
    quotes = [
        "The best way to get started is to quit talking and begin doing. - Walt Disney",
        "The pessimist sees difficulty in every opportunity. The optimist sees opportunity in every difficulty. - Winston Churchill",
        "Don’t let yesterday take up too much of today. - Will Rogers",
        "You learn more from failure than from success. Don’t let it stop you. Failure builds character. - Unknown",
    ]
    return random.choice(quotes)

@mcp.tool()
def get_inspirational_message() -> str:
    """Provide a random inspirational message"""
    messages = [
        "Believe you can and you're halfway there. - Theodore Roosevelt",
        "Act as if what you do makes a difference. It does. - William James",
        "Success is not final, failure is not fatal: It is the courage to continue that counts. - Winston Churchill",
        "What lies behind us and what lies before us are tiny matters compared to what lies within us. - Ralph Waldo Emerson",
    ]
    return random.choice(messages)

if __name__ == "__main__":
    mcp.run("stdio")