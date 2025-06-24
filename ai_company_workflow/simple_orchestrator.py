import asyncio
import json
from typing import Dict, List, Any
import uuid
from a2a_client import A2AToolClient

class SimpleOrchestrator:
    """A simple orchestrator that coordinates between A2A agents using the A2A client"""
    
    def __init__(self):
        self.agents = {
            "research_analyst": "http://localhost:10031",
            "data_processor": "http://localhost:10032", 
            "report_writer": "http://localhost:10033"
        }
        
        # Initialize A2A client
        self.a2a_client = A2AToolClient()
        
        # Register all agents with the client
        for name, url in self.agents.items():
            self.a2a_client.add_remote_agent(url)
            print(f"✅ Registered {name} at {url}")
    
    async def send_message_to_agent(self, agent_url: str, message: str) -> Dict[str, Any]:
        """Send a message to an A2A agent using the A2A client"""
        try:
            result = await self.a2a_client.create_task(agent_url, message)
            return {"success": True, "content": result}
        except Exception as e:
            return {"error": f"Failed to communicate with agent: {str(e)}"}
    
    def extract_text_from_response(self, response: Dict[str, Any]) -> str:
        """Extract text content from an A2A response"""
        try:
            if "error" in response:
                return f"Error: {response['error']}"
            
            if "content" in response:
                return response["content"]
            
            return "No content found in response"
        except Exception as e:
            return f"Error extracting content: {str(e)}"
    
    async def orchestrate_workflow(self, user_request: str) -> str:
        """Main orchestration workflow"""
        try:
            # Step 1: Research Phase
            print("🔄 Step 1: Researching...")
            research_message = f"Research the following topic thoroughly: {user_request}"
            research_response = await self.send_message_to_agent(
                self.agents["research_analyst"], 
                research_message
            )
            research_content = self.extract_text_from_response(research_response)
            
            if "Error:" in research_content:
                return f"❌ Research failed: {research_content}"
            
            print("✅ Research completed")
            print(f"📄 Research content length: {len(research_content)} characters")
            
            # Step 2: Data Processing Phase
            print("🔄 Step 2: Processing data...")
            data_message = f"Analyze and process this research data: {research_content[:2000]}..."  # Truncate if too long
            data_response = await self.send_message_to_agent(
                self.agents["data_processor"], 
                data_message
            )
            data_content = self.extract_text_from_response(data_response)
            
            if "Error:" in data_content:
                return f"❌ Data processing failed: {data_content}"
            
            print("✅ Data processing completed")
            print(f"📊 Data content length: {len(data_content)} characters")
            
            # Step 3: Report Generation Phase
            print("🔄 Step 3: Generating report...")
            report_message = f"""Create a comprehensive report based on the following information:

RESEARCH FINDINGS:
{research_content[:1500]}...

DATA ANALYSIS:
{data_content[:1500]}...

Please create a well-structured report that combines both the research findings and data analysis."""
            
            report_response = await self.send_message_to_agent(
                self.agents["report_writer"], 
                report_message
            )
            report_content = self.extract_text_from_response(report_response)
            
            if "Error:" in report_content:
                return f"❌ Report generation failed: {report_content}"
            
            print("✅ Report generation completed")
            print(f"📋 Report content length: {len(report_content)} characters")
            
            # Step 4: Compile final result
            final_result = f"""# Workflow Orchestration Complete

## Summary
Successfully coordinated between Research Analyst, Data Processor, and Report Writer agents using A2A protocol.

## Final Report
{report_content}

---
*This report was generated through coordinated work between specialized AI agents using the A2A protocol.*"""
            
            return final_result
            
        except Exception as e:
            return f"❌ Orchestration failed: {str(e)}"
    
    def get_agent_status(self) -> Dict[str, str]:
        """Check if all agents are available"""
        status = {}
        for name, url in self.agents.items():
            try:
                import requests
                response = requests.get(f"{url}/.well-known/agent.json", timeout=5)
                if response.status_code == 200:
                    agent_data = response.json()
                    status[name] = f"✅ Available - {agent_data.get('name', 'Unknown')}"
                else:
                    status[name] = "❌ Not responding"
            except Exception as e:
                status[name] = f"❌ Unreachable - {str(e)}"
        return status
    
    def list_available_agents(self) -> List[Dict[str, Any]]:
        """List all available agents using the A2A client"""
        try:
            return self.a2a_client.list_remote_agents()
        except Exception as e:
            print(f"Error listing agents: {e}")
            return []

# Create the orchestrator instance
orchestrator = SimpleOrchestrator()

# Test function
async def test_orchestrator():
    """Test the orchestrator with a simple request"""
    print("🧪 Testing Simple Orchestrator with A2A Client...")
    
    # Check agent status
    print("\n📊 Agent Status:")
    status = orchestrator.get_agent_status()
    for agent, status_text in status.items():
        print(f"  {agent}: {status_text}")
    
    # List available agents via A2A client
    print("\n🔍 Available Agents (via A2A client):")
    agents = orchestrator.list_available_agents()
    for agent in agents:
        print(f"  - {agent.get('name', 'Unknown')}: {agent.get('description', 'No description')}")
    
    # Test orchestration
    print("\n🚀 Testing workflow orchestration...")
    result = await orchestrator.orchestrate_workflow("AI trends in 2024")
    print("\n📋 Final Result:")
    print(result)

if __name__ == "__main__":
    asyncio.run(test_orchestrator()) 