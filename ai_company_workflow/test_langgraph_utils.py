#!/usr/bin/env python3
"""
Test script for langgraph_server_utils
This script tests the new langgraph_server_utils to ensure it can create A2A servers for LangGraph agents.
"""

import asyncio
import uvicorn
import threading
import time
from langgraph_server_utils import create_data_processor_server, create_langgraph_agent_a2a_server
from langgraph_data_processor import LangGraphDataProcessor
from a2a.types import AgentSkill

def test_create_data_processor_server():
    """Test creating a data processor server using langgraph_server_utils."""
    print("🧪 Testing create_data_processor_server...")
    
    try:
        server = create_data_processor_server("localhost", 10032)
        print("✅ Data processor server created successfully")
        print(f"📋 Server type: {type(server).__name__}")
        return server
    except Exception as e:
        print(f"❌ Failed to create data processor server: {e}")
        return None

def test_create_custom_agent_server():
    """Test creating a custom agent server using the generic function."""
    print("\n🧪 Testing create_langgraph_agent_a2a_server with custom agent...")
    
    # Define custom skills for testing
    custom_skills = [
        AgentSkill(
            id='custom_analysis',
            name='Custom Analysis',
            description='A custom analysis capability for testing',
            tags=['custom', 'analysis', 'test'],
            examples=['Analyze this custom data', 'Process this test content']
        )
    ]
    
    try:
        server = create_langgraph_agent_a2a_server(
            agent_class=LangGraphDataProcessor,
            name="Custom Test Agent",
            description="A custom test agent created with langgraph_server_utils",
            skills=custom_skills,
            host="localhost",
            port=10036
        )
        print("✅ Custom agent server created successfully")
        print(f"📋 Server type: {type(server).__name__}")
        return server
    except Exception as e:
        print(f"❌ Failed to create custom agent server: {e}")
        return None

def run_server_in_background(server, port, name):
    """Run a server in the background."""
    def run_server():
        try:
            uvicorn.run(server.build(), host="localhost", port=port, log_level="error")
        except Exception as e:
            print(f"❌ Error running {name} server: {e}")
    
    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()
    return thread

def test_server_functionality(port, name):
    """Test if a server is responding correctly."""
    import requests
    import time
    
    # Wait for server to start
    time.sleep(3)
    
    try:
        # Test agent metadata
        response = requests.get(f"http://localhost:{port}/.well-known/agent.json", timeout=5)
        if response.status_code == 200:
            agent_data = response.json()
            print(f"✅ {name} server is responding")
            print(f"📝 Agent name: {agent_data.get('name', 'Unknown')}")
            print(f"🔧 Skills: {[skill.get('name') for skill in agent_data.get('skills', [])]}")
            return True
        else:
            print(f"❌ {name} server not responding (status: {response.status_code})")
            return False
    except Exception as e:
        print(f"❌ {name} server not reachable: {e}")
        return False

def main():
    """Main test function."""
    print("🚀 Testing langgraph_server_utils...")
    
    # Test 1: Create data processor server
    data_server = test_create_data_processor_server()
    
    # Test 2: Create custom agent server
    custom_server = test_create_custom_agent_server()
    
    if data_server and custom_server:
        print("\n🧪 Testing server functionality...")
        
        # Start servers in background
        data_thread = run_server_in_background(data_server, 10032, "Data Processor")
        custom_thread = run_server_in_background(custom_server, 10036, "Custom Agent")
        
        # Test server functionality
        data_working = test_server_functionality(10032, "Data Processor")
        custom_working = test_server_functionality(10036, "Custom Agent")
        
        print("\n📊 Test Results:")
        print(f"Data Processor Server: {'✅ Working' if data_working else '❌ Failed'}")
        print(f"Custom Agent Server: {'✅ Working' if custom_working else '❌ Failed'}")
        
        if data_working and custom_working:
            print("\n🎉 All tests passed! langgraph_server_utils is working correctly.")
        else:
            print("\n⚠️ Some tests failed. Check the logs above.")
    else:
        print("\n❌ Server creation tests failed. Cannot test functionality.")

if __name__ == "__main__":
    main()