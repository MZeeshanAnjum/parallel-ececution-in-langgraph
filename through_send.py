import asyncio
from typing import List, Dict, Annotated
from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

def override(existing, new):
    """Properly merge lists from parallel execution"""
    if existing is None:
        existing = []
    if new is None:
        new = []
    
    # Extend existing list with new messages
    result = existing.copy()
    result.extend(new)
    return result

class TestState(BaseModel):
    query: str = ""
    messages: Annotated[List[Dict], override] = []
    internal_messages: Annotated[List[Dict], override] = []
    parallel_intents: bool = False
    intent_1: bool = False
    intent_2: bool = False
    intent_3: bool = False

class TestAgent:
    
    def classify_intents(self, state: TestState) -> TestState:
        """Simulate intent classification"""
        print(f"🔍 Classifying query: '{state.query}'")
        
        # Simple intent detection based on keywords
        intent_1 = "team" in state.query.lower()
        intent_2 = "performance" in state.query.lower() 
        intent_3 = "status" in state.query.lower()
        
        # Check if multiple intents detected
        active_intents = sum([intent_1, intent_2, intent_3])
        parallel_intents = active_intents > 1
        
        print(f"📊 Detected intents: team={intent_1}, performance={intent_2}, status={intent_3}")
        print(f"🔄 Parallel execution needed: {parallel_intents}")
        
        return {
            "intent_1": intent_1,
            "intent_2": intent_2, 
            "intent_3": intent_3,
            "parallel_intents": parallel_intents
        }
    
    def dispatch_parallel_tasks(self, state: TestState):
        """Dispatch parallel tasks using Send API"""
        print("🚀 Dispatching parallel tasks...")
        
        sends = []
        
        if state.intent_1:
            sends.append(Send("task_1", state))
            print("   📤 Sending to task_1")
            
        if state.intent_2:
            sends.append(Send("task_2", state))
            print("   📤 Sending to task_2")
            
        if state.intent_3:
            sends.append(Send("task_3", state))
            print("   📤 Sending to task_3")
        
        return sends
    
    async def task_1(self, state: TestState) -> TestState:
        """Simulate team information task"""
        print("⚙️  Task 1: Processing team information...")
        await asyncio.sleep(1)  # Simulate work
        
        msg = {
            "role": "Team Assistant",
            "content": f"Team info for query: '{state.query}'"
        }
        
        print("✅ Task 1 completed")
        return {"internal_messages": [msg]}
    
    async def task_2(self, state: TestState) -> TestState:
        """Simulate performance analysis task"""
        print("⚙️  Task 2: Processing performance analysis...")
        await asyncio.sleep(1.5)  # Simulate work
        
        msg = {
            "role": "Performance Analyzer", 
            "content": f"Performance data for query: '{state.query}'"
        }
        
        print("✅ Task 2 completed")
        return {"internal_messages": [msg]}
    
    async def task_3(self, state: TestState) -> TestState:
        """Simulate status check task"""
        print("⚙️  Task 3: Processing status check...")
        await asyncio.sleep(0.8)  # Simulate work
        
        msg = {
            "role": "Status Checker",
            "content": f"Status update for query: '{state.query}'"
        }
        
        print("✅ Task 3 completed")
        return {"internal_messages": [msg]}
    
    def aggregate_results(self, state: TestState) -> TestState:
        """Aggregate all parallel results"""
        print("📋 Aggregating parallel results...")
        print(f"   📨 Total messages collected: {len(state.internal_messages)}")
        
        for i, msg in enumerate(state.internal_messages, 1):
            print(f"   {i}. {msg['role']}: {msg['content']}")
        
        return {}
    
    def generate_response(self, state: TestState) -> TestState:
        """Generate final response"""
        print("📝 Generating final response...")
        
        # Combine all internal messages into a response
        response_parts = []
        for msg in state.internal_messages:
            response_parts.append(f"- From {msg['role']}: {msg['content']}")
        
        combined_response = "Here's what I found:\n" + "\n".join(response_parts)
        
        final_msg = {
            "role": "assistant",
            "content": combined_response
        }
        
        messages = state.messages.copy()
        messages.append(final_msg)
        
        return {"messages": messages}
    
    def route_intents(self, state: TestState):
        """Route based on intents detected"""
        if state.parallel_intents:
            # Return the Send objects directly
            return self.dispatch_parallel_tasks(state)
        elif state.intent_1:
            return "task_1"
        elif state.intent_2:
            return "task_2"
        elif state.intent_3:
            return "task_3"
        else:
            return "generate_response"
    
    def build_graph(self):
        """Build the graph with Send API"""
        builder = StateGraph(TestState)
        
        # Add nodes
        builder.add_node("classify_intents", self.classify_intents)
        builder.add_node("task_1", self.task_1)
        builder.add_node("task_2", self.task_2) 
        builder.add_node("task_3", self.task_3)
        builder.add_node("aggregate_results", self.aggregate_results)
        builder.add_node("generate_response", self.generate_response)
        
        # Add edges
        builder.add_edge(START, "classify_intents")
        
        # Conditional routing - remove the dispatch_parallel mapping
        builder.add_conditional_edges("classify_intents", self.route_intents, {
            "task_1": "task_1",
            "task_2": "task_2", 
            "task_3": "task_3",
            "generate_response": "generate_response"
        })
        
        # All parallel tasks automatically converge to aggregate_results
        builder.add_edge(["task_1", "task_2", "task_3"], "aggregate_results")
        # builder.add_edge("aggregate_results", "generate_response")
        builder.add_edge("generate_response", END)
        
        return builder.compile()

# Test the Send API example
async def test_send_api():
    print("🧪 Testing Send API Example")
    print("=" * 50)
    
    agent = TestAgent()
    graph = agent.build_graph()
    
    # Test cases
    test_queries = [
        "Show me team performance status",  # Should trigger all 3 (parallel)
        "What's the team information?",     # Should trigger only task_1
        "Check performance metrics",        # Should trigger only task_2
        "Give me current status",          # Should trigger only task_3
        "Hello there"                      # Should trigger none
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔥 Test Case {i}: '{query}'")
        print("-" * 40)
        
        initial_state = {
            "query": query,
            "messages": [{"role": "user", "content": query}]
        }
        
        result = await graph.ainvoke(initial_state)
        
        print(f"📤 Final Response:")
        if result["messages"]:
            print(f"   {result['messages'][-1]['content']}")
        
        print(f"📊 Internal Messages: {len(result.get('internal_messages', []))}")
        print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_send_api())
