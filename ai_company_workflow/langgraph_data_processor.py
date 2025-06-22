import os
from collections.abc import AsyncIterable
from typing import Any, Literal, Dict, List
import json
import re

from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
import httpx

memory = MemorySaver()

# Define the state schema for our data processing workflow
class DataProcessingState(BaseModel):
    """State for data processing workflow."""
    messages: List[Any] = Field(default_factory=list)
    data_input: str = ""
    analysis_type: str = ""
    extracted_entities: List[str] = Field(default_factory=list)
    key_insights: List[str] = Field(default_factory=list)
    summary: str = ""
    final_report: str = ""

class ResponseFormat(BaseModel):
    """Response format for the data processing agent."""
    status: Literal['input_required', 'completed', 'error'] = 'input_required'
    message: str
    analysis_type: str = ""
    insights: List[str] = Field(default_factory=list)

@tool
def extract_entities(text: str) -> Dict[str, Any]:
    """Extract named entities (companies, people, locations, dates) from text.
    
    Args:
        text: The text to analyze for entities
        
    Returns:
        Dictionary containing extracted entities by category
    """
    try:
        # Simple entity extraction using regex patterns
        entities = {
            "companies": [],
            "people": [],
            "locations": [],
            "dates": [],
            "numbers": []
        }
        
        # Extract company names (words starting with capital letters, possibly with Inc, Corp, etc.)
        company_pattern = r'\b[A-Z][a-zA-Z\s&]+(?:Inc|Corp|LLC|Ltd|Company|Co|Corporation)\b'
        companies = re.findall(company_pattern, text, re.IGNORECASE)
        entities["companies"] = list(set(companies))
        
        # Extract people names (two consecutive capitalized words)
        people_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        people = re.findall(people_pattern, text)
        entities["people"] = list(set(people))
        
        # Extract locations (words ending with common location suffixes)
        location_pattern = r'\b[A-Z][a-zA-Z\s]+(?:City|State|Country|Street|Avenue|Road|Park)\b'
        locations = re.findall(location_pattern, text, re.IGNORECASE)
        entities["locations"] = list(set(locations))
        
        # Extract dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b\d{4}-\d{2}-\d{2}\b|\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        dates = re.findall(date_pattern, text, re.IGNORECASE)
        entities["dates"] = list(set(dates))
        
        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?(?:%|million|billion|thousand)?\b'
        numbers = re.findall(number_pattern, text, re.IGNORECASE)
        entities["numbers"] = list(set(numbers))
        
        return entities
    except Exception as e:
        return {"error": f"Failed to extract entities: {str(e)}"}

@tool
def analyze_sentiment(text: str) -> Dict[str, Any]:
    """Analyze the sentiment of the provided text.
    
    Args:
        text: The text to analyze for sentiment
        
    Returns:
        Dictionary containing sentiment analysis results
    """
    try:
        # Simple sentiment analysis based on keyword matching
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'positive', 'success', 'growth', 'increase', 'improve', 'benefit', 'advantage']
        negative_words = ['bad', 'terrible', 'awful', 'negative', 'failure', 'decline', 'decrease', 'problem', 'issue', 'risk', 'concern', 'worry']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            sentiment = "positive"
            confidence = min(0.9, (positive_count - negative_count) / max(positive_count + negative_count, 1))
        elif negative_count > positive_count:
            sentiment = "negative"
            confidence = min(0.9, (negative_count - positive_count) / max(positive_count + negative_count, 1))
        else:
            sentiment = "neutral"
            confidence = 0.5
            
        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 2),
            "positive_score": positive_count,
            "negative_score": negative_count
        }
    except Exception as e:
        return {"error": f"Failed to analyze sentiment: {str(e)}"}

@tool
def summarize_text(text: str, max_length: int = 200) -> str:
    """Create a concise summary of the provided text.
    
    Args:
        text: The text to summarize
        max_length: Maximum length of the summary
        
    Returns:
        A concise summary of the text
    """
    try:
        # Simple summarization by extracting key sentences
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Score sentences based on word frequency and position
        word_freq = {}
        for sentence in sentences:
            words = re.findall(r'\b\w+\b', sentence.lower())
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_freq[word] = word_freq.get(word, 0) + 1
        
        # Score sentences
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            words = re.findall(r'\b\w+\b', sentence.lower())
            score = sum(word_freq.get(word, 0) for word in words if len(word) > 3)
            # Bonus for first few sentences
            if i < 3:
                score *= 1.5
            sentence_scores.append((score, sentence))
        
        # Get top sentences
        sentence_scores.sort(reverse=True)
        summary_sentences = []
        current_length = 0
        
        for score, sentence in sentence_scores:
            if current_length + len(sentence) <= max_length:
                summary_sentences.append(sentence)
                current_length += len(sentence)
            else:
                break
        
        summary = '. '.join(summary_sentences) + '.'
        return summary if summary else text[:max_length] + "..."
    except Exception as e:
        return f"Failed to summarize text: {str(e)}"

@tool
def categorize_content(text: str) -> Dict[str, Any]:
    """Categorize the content based on keywords and topics.
    
    Args:
        text: The text to categorize
        
    Returns:
        Dictionary containing content categories and confidence scores
    """
    try:
        categories = {
            "technology": ["ai", "artificial intelligence", "machine learning", "software", "hardware", "computer", "digital", "tech", "algorithm", "data"],
            "business": ["company", "business", "market", "revenue", "profit", "investment", "strategy", "management", "corporate", "financial"],
            "health": ["health", "medical", "medicine", "patient", "treatment", "disease", "healthcare", "clinical", "therapy", "diagnosis"],
            "science": ["research", "study", "scientific", "experiment", "discovery", "theory", "hypothesis", "analysis", "laboratory", "methodology"],
            "finance": ["money", "finance", "banking", "investment", "stock", "market", "economy", "financial", "currency", "trading"]
        }
        
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                scores[category] = min(1.0, score / len(keywords))
        
        # Sort by score
        sorted_categories = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "primary_category": sorted_categories[0][0] if sorted_categories else "general",
            "categories": dict(sorted_categories),
            "confidence": sorted_categories[0][1] if sorted_categories else 0.0
        }
    except Exception as e:
        return {"error": f"Failed to categorize content: {str(e)}"}

class LangGraphDataProcessor:
    """Advanced Data Processing Agent using LangGraph for sophisticated workflows."""

    SYSTEM_INSTRUCTION = (
        'You are an advanced data processing agent that can analyze text data using multiple tools. '
        'You can extract entities, analyze sentiment, summarize content, and categorize information. '
        'Use the available tools to provide comprehensive analysis of the input data. '
        'Always provide structured, actionable insights based on your analysis.'
    )

    FORMAT_INSTRUCTION = (
        'Set response status to input_required if the user needs to provide more information. '
        'Set response status to error if there is an error while processing. '
        'Set response status to completed when the analysis is complete. '
        'Include the analysis type and key insights in your response.'
    )

    def __init__(self):
        # Only use Google Gemini model
        self.model = ChatGoogleGenerativeAI(model='gemini-2.0-flash')
        self.tools = [extract_entities, analyze_sentiment, summarize_text, categorize_content]

        # Create the LangGraph agent
        self.graph = create_react_agent(
            self.model,
            tools=self.tools,
            checkpointer=memory,
            prompt=self.SYSTEM_INSTRUCTION,
            response_format=(self.FORMAT_INSTRUCTION, ResponseFormat),
        )

    async def stream(self, query: str, context_id: str) -> AsyncIterable[dict[str, Any]]:
        """Stream the data processing workflow."""
        inputs = {'messages': [('user', query)]}
        config = {'configurable': {'thread_id': context_id}}

        for item in self.graph.stream(inputs, config, stream_mode='values'):
            message = item['messages'][-1]
            if (
                isinstance(message, AIMessage)
                and message.tool_calls
                and len(message.tool_calls) > 0
            ):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Processing data with advanced analysis tools...',
                }
            elif isinstance(message, ToolMessage):
                yield {
                    'is_task_complete': False,
                    'require_user_input': False,
                    'content': 'Analyzing data patterns and extracting insights...',
                }

        yield self.get_agent_response(config)

    def get_agent_response(self, config):
        """Get the final response from the agent."""
        current_state = self.graph.get_state(config)
        structured_response = current_state.values.get('structured_response')
        
        if structured_response and isinstance(structured_response, ResponseFormat):
            if structured_response.status == 'input_required':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'error':
                return {
                    'is_task_complete': False,
                    'require_user_input': True,
                    'content': structured_response.message,
                }
            if structured_response.status == 'completed':
                return {
                    'is_task_complete': True,
                    'require_user_input': False,
                    'content': structured_response.message,
                }

        return {
            'is_task_complete': False,
            'require_user_input': True,
            'content': (
                'We are unable to process your request at the moment. '
                'Please try again.'
            ),
        }

    SUPPORTED_CONTENT_TYPES = ['text', 'text/plain'] 