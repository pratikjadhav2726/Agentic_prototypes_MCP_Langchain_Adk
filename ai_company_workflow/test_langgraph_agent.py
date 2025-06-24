#!/usr/bin/env python3
"""
Test script for the LangGraph Data Processor Agent
This script tests the new LangGraph-based data processing agent to ensure it works correctly with the A2A protocol.
"""

import asyncio
import json
import requests
from a2a_client import A2AToolClient

async def test_langgraph_agent():
    """Test the LangGraph data processor agent"""
    print("🧪 Testing LangGraph Data Processor Agent...")
    
    # Initialize A2A client
    a2a_client = A2AToolClient()
    agent_url = "http://localhost:10032"
    a2a_client.add_remote_agent(agent_url)
    
    # Test 1: Check if agent is available
    print("\n📊 Checking agent availability...")
    try:
        response = requests.get(f"{agent_url}/.well-known/agent.json", timeout=5)
        if response.status_code == 200:
            agent_data = response.json()
            print(f"✅ Agent is available: {agent_data.get('name', 'Unknown')}")
            print(f"📝 Description: {agent_data.get('description', 'No description')}")
            print(f"🔧 Skills: {[skill.get('name') for skill in agent_data.get('skills', [])]}")
        else:
            print(f"❌ Agent not responding: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot reach agent: {e}")
        return
    
    # Test 2: Test entity extraction
    print("\n🔍 Test 1: Entity Extraction")
    test_text = """
    Apple Inc. announced record profits in Q4 2023, with CEO Tim Cook stating that 
    the company's new AI initiatives in Cupertino, California are driving growth. 
    The company reported $119.6 billion in revenue, up 8% from last year. 
    Microsoft Corporation also saw strong performance with Satya Nadella leading 
    their cloud computing division to success.
    """
    
    entity_message = f"Extract all entities from this text: {test_text}"
    try:
        result = await a2a_client.create_task(agent_url, entity_message)
        print("✅ Entity extraction completed")
        print(f"📄 Result: {result[:500]}...")
    except Exception as e:
        print(f"❌ Entity extraction failed: {e}")
    
    # Test 3: Test sentiment analysis
    print("\n😊 Test 2: Sentiment Analysis")
    sentiment_text = "This is an excellent product with amazing features and great customer service!"
    
    sentiment_message = f"Analyze the sentiment of this text: {sentiment_text}"
    try:
        result = await a2a_client.create_task(agent_url, sentiment_message)
        print("✅ Sentiment analysis completed")
        print(f"📄 Result: {result[:500]}...")
    except Exception as e:
        print(f"❌ Sentiment analysis failed: {e}")
    
    # Test 4: Test content categorization
    print("\n🏷️ Test 3: Content Categorization")
    tech_text = """
    Artificial intelligence and machine learning are transforming the technology industry. 
    Companies are investing heavily in AI research and development to gain competitive advantages. 
    The latest algorithms show promising results in natural language processing tasks.
    """
    
    category_message = f"Categorize this content: {tech_text}"
    try:
        result = await a2a_client.create_task(agent_url, category_message)
        print("✅ Content categorization completed")
        print(f"📄 Result: {result[:500]}...")
    except Exception as e:
        print(f"❌ Content categorization failed: {e}")
    
    # Test 5: Test summarization
    print("\n📝 Test 4: Text Summarization")
    long_text = """
    The field of artificial intelligence has seen remarkable progress in recent years. 
    Machine learning algorithms have become more sophisticated, enabling computers to 
    perform tasks that were once thought to be exclusively human. Natural language 
    processing has advanced significantly, with models like GPT-4 demonstrating 
    unprecedented capabilities in understanding and generating human-like text. 
    Computer vision has also made great strides, with applications ranging from 
    autonomous vehicles to medical imaging. The integration of AI into various 
    industries is creating new opportunities and challenges for businesses and 
    society as a whole. Companies are investing billions of dollars in AI research 
    and development, recognizing the potential for competitive advantages and 
    operational efficiencies.
    """
    
    summary_message = f"Summarize this text: {long_text}"
    try:
        result = await a2a_client.create_task(agent_url, summary_message)
        print("✅ Text summarization completed")
        print(f"📄 Result: {result[:500]}...")
    except Exception as e:
        print(f"❌ Text summarization failed: {e}")
    
    # Test 6: Test comprehensive analysis
    print("\n🔬 Test 5: Comprehensive Analysis")
    comprehensive_text = """
    Tesla Inc. reported strong quarterly results with CEO Elon Musk announcing 
    significant progress in autonomous driving technology. The company's revenue 
    increased by 15% to $25.2 billion, driven by strong demand for electric vehicles 
    in North America and Europe. Tesla's AI-powered Full Self-Driving system 
    continues to improve, with new features being rolled out to customers in 
    California and Texas. The company plans to expand its manufacturing facilities 
    in Austin, Texas and Berlin, Germany to meet growing demand.
    """
    
    analysis_message = f"Perform a comprehensive analysis of this text including entity extraction, sentiment analysis, categorization, and summarization: {comprehensive_text}"
    try:
        result = await a2a_client.create_task(agent_url, analysis_message)
        print("✅ Comprehensive analysis completed")
        print(f"📄 Result: {result[:800]}...")
    except Exception as e:
        print(f"❌ Comprehensive analysis failed: {e}")
    
    print("\n🎉 LangGraph Data Processor Agent testing completed!")

if __name__ == "__main__":
    asyncio.run(test_langgraph_agent()) 