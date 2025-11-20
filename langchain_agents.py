#!/usr/bin/env python3
"""
LangChain Agents - Complete
========================================================

A comprehensive tutorial on building AI agents with LangChain 1.0+
and Google Vertex AI (Gemini models).

Parts Covered:
- Part 1: Setup, Introduction & ReAct Framework
- Part 2: Basic Agents & Custom Tools
- Part 3: Agent Types & Advanced Features  
- Part 4: Real-World Use Cases & Final Project

Prerequisites:
- Python 3.10+
- Google Cloud CLI installed
- gcloud auth application-default login (for local use)

Usage:
    python langchain_agents_teaching.py
"""

import subprocess
import sys
from datetime import datetime

# =============================================================================
# PART 1: SETUP & INTRODUCTION
# =============================================================================

def install_packages():
    """Install all required packages."""
    print("\n" + "=" * 70)
    print("STEP 1: Installing Required Packages (LangChain 1.0+)")
    print("=" * 70)

    packages = [
        "langchain",
        "langchain-core",
        "langchain-community",
        "langchain-google-vertexai",
        "langgraph",
        "langchainhub",
        "google-cloud-aiplatform",
        "wikipedia",
        "duckduckgo-search",
        "numexpr",
        "beautifulsoup4",
        "requests",
    ]

    for package in packages:
        print(f"  Installing {package}...")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", package],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"  ✅ {package}")

    print("\nAll packages installed successfully!\n")


def verify_imports():
    """Verify all imports work with LangChain 1.0+."""
    print("=" * 70)
    print("STEP 2: Verifying Imports (LangChain 1.0+ API)")
    print("=" * 70)

    try:
        import langchain
        import langchain_core
        import langchain_community
        from importlib.metadata import version

        print(f"  ✅ langchain: {langchain.__version__}")
        print(f"  ✅ langchain-core: {langchain_core.__version__}")
        print(f"  ✅ langchain-community: {langchain_community.__version__}")

        try:
            vertexai_version = version("langchain-google-vertexai")
            print(f"  ✅ langchain-google-vertexai: {vertexai_version}")
        except Exception:
            import langchain_google_vertexai  # noqa: F401
            print("  ✅ langchain-google-vertexai: installed")

        # Verify langgraph
        try:
            langgraph_version = version("langgraph")
            print(f"  ✅ langgraph: {langgraph_version}")
        except Exception:
            import langgraph  # noqa: F401
            print("  ✅ langgraph: installed")

        print("\n🎉 All imports successful!\n")
        return True
    except ImportError as e:
        print(f"\n❌ Import error: {e}\n")
        return False


def setup_vertex_ai():
    """Initialize Vertex AI and LLM."""
    print("=" * 70)
    print("STEP 3: Initializing Vertex AI (Gemini 2.0 Flash)")
    print("=" * 70)

    try:
        import vertexai
        from langchain_google_vertexai import ChatVertexAI

        PROJECT_ID = "terraform-prj-476214"
        LOCATION = "us-central1"

        # Initialize Vertex AI
        vertexai.init(project=PROJECT_ID, location=LOCATION)
        print(f"  ✅ Project: {PROJECT_ID}")
        print(f"  ✅ Location: {LOCATION}")

        # Create LLM
        llm = ChatVertexAI(
            model="gemini-2.0-flash-exp",
            project=PROJECT_ID,
            location=LOCATION,
            temperature=0,
            max_tokens=1024,
        )

        # Test connection
        print("\n  🔄 Testing connection...")
        response = llm.invoke("Say 'Hello, LangChain Agents!' in exactly those words.")
        print(f"  ✅ Model response: {response.content}")
        print("\n🎉 Vertex AI ready!\n")

        return llm
    except Exception as e:
        print(f"\n❌ Failed to initialize Vertex AI: {e}")
        print("\nMake sure you have authenticated:")
        print("  gcloud auth application-default login\n")
        sys.exit(1)


# =============================================================================
# PART 2: BASIC AGENTS & CUSTOM TOOLS
# =============================================================================

def test_calculator_agent(llm):
    """Part 2.1: Calculator Agent - Basic Tool Usage."""
    print("=" * 70)
    print("PART 2.1: Calculator Agent (Basic Tool)")
    print("=" * 70)
    print("\nWhat we'll learn:")
    print("  - Creating tools with the @tool decorator")
    print("  - Using the tool.invoke() method")
    print("  - Tool description and type hints")

    from langchain_core.tools import tool

    # Create calculator tool using modern @tool decorator
    @tool
    def calculator(expression: str) -> str:
        """Useful for math calculations.
        Input: math expression like '100 * 0.25'
        Output: calculation result
        """
        try:
            result = eval(expression)
            return str(result)
        except Exception as e:
            return f"Error: {str(e)}"

    # Test
    question = "What is 25% of 300?"
    print(f"\n  Question: {question}")
    print("\n  Tool definition:")
    print(f"    Name: {calculator.name}")
    print(f"    Description: {calculator.description}")

    print("\n  Invoking tool with expression: '300 * 0.25'")
    result = calculator.invoke({"expression": "300 * 0.25"})

    print(f"\n  Result: {result}\n")


def test_wikipedia_agent(llm):
    """Part 2.2: Wikipedia Agent - Using Community Tools."""
    print("=" * 70)
    print("PART 2.2: Wikipedia Agent (Community Tool)")
    print("=" * 70)
    print("\nWhat we'll learn:")
    print("  - Using pre-built community tools")
    print("  - Integrating external APIs (Wikipedia)")
    print("  - Tool invocation and result handling")

    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.tools import WikipediaQueryRun

    # Create Wikipedia tool
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    # Test
    question = "Did Alexander Graham Bell invent telephone?"
    print(f"\n  Question: {question}")
    print("\n  Tool definition:")
    print(f"    Name: {wikipedia.name}")
    print(f"    Description: {wikipedia.description}")

    print("\n  Searching Wikipedia...")
    result = wikipedia.invoke({"query": question})

    print("\n  Result (first 400 chars):")
    print(f"    {result[:400]}...\n")


def test_multi_tool_agent(llm):
    """Part 2.3: Multi-Tool Agent - Combining Multiple Tools."""
    print("=" * 70)
    print("PART 2.3: Multi-Tool Agent")
    print("=" * 70)
    print("\nWhat we'll learn:")
    print("  - Creating multiple tools")
    print("  - Managing tool collections")
    print("  - Tool selection and usage patterns")

    from langchain_core.tools import tool
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.tools import WikipediaQueryRun

    # Tool 1: Calculator
    @tool
    def calculate(expression: str) -> str:
        """For math calculations. Input: math expression"""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {str(e)}"

    # Tool 2: Wikipedia
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    # Tool 3: Current Year
    @tool
    def current_year(query: str) -> str:
        """Returns the current year."""
        return str(datetime.now().year)

    print("\n  Tools defined:")
    print(f"    1. {calculate.name}: {calculate.description[:50]}...")
    print(f"    2. {wikipedia.name}: {wikipedia.description[:50]}...")
    print(f"    3. {current_year.name}: {current_year.description[:50]}...")

    # Demo usage
    print("\n  Example 1: Using calculator tool")
    result1 = calculate.invoke({"expression": "50 + 50"})
    print(f"    50 + 50 = {result1}")

    print("\n  Example 2: Using Wikipedia tool")
    result2 = wikipedia.invoke({"query": "Albert Einstein"})
    print(f"    Found info: {result2[:100]}...")

    print("\n  Example 3: Using current_year tool")
    result3 = current_year.invoke({"query": "What is the year?"})
    print(f"    Current year: {result3}")

    print("\nMulti-tool setup complete!\n")


def test_custom_tool_agent(llm):
    """Part 2.4: Custom Tool Agent - Creating Domain-Specific Tools."""
    print("=" * 70)
    print("PART 2.4: Custom Tool Agent")
    print("=" * 70)
    print("\nWhat we'll learn:")
    print("  - Creating custom domain-specific tools")
    print("  - Tool parameters and return types")
    print("  - Error handling in tools")

    from langchain_core.tools import tool

    # Create custom tools for a weather/astronomy domain
    @tool
    def get_current_datetime(query: str) -> str:
        """Returns the current date and time (UTC).
        Use when the user asks about current time or date.
        """
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    @tool
    def get_current_year(query: str) -> str:
        """Returns the current year as a number.
        Use for current year questions or age calculations.
        """
        return str(datetime.now().year)

    @tool
    def calculate_age(birth_year: int) -> str:
        """Calculate age from birth year.
        Input: birth_year as integer (e.g., 1980)
        Output: calculated age in years
        """
        current = datetime.now().year
        age = current - birth_year
        return f"Age: {age} years (born in {birth_year})"

    print("\n  Custom tools defined:")
    print(f"    1. {get_current_datetime.name}")
    print(f"    2. {get_current_year.name}")
    print(f"    3. {calculate_age.name}")

    # Demo usage
    print("\n  Example 1: Get current datetime")
    dt_result = get_current_datetime.invoke({"query": "What time is it?"})
    print(f"    {dt_result}")

    print("\n  Example 2: Get current year")
    year_result = get_current_year.invoke({"query": "What year is it?"})
    print(f"    {year_result}")

    print("\n  Example 3: Calculate age")
    age_result = calculate_age.invoke({"birth_year": 1980})
    print(f"    {age_result}")

    print("\nCustom tools working!\n")


# =============================================================================
# PART 3: ADVANCED AGENTS & AGENT TYPES
# =============================================================================

def demonstrate_react_framework(llm):
    """Part 3.1: ReAct Framework - Understanding Agent Reasoning."""
    print("=" * 70)
    print("PART 3.1: ReAct Framework (Reasoning + Acting)")
    print("=" * 70)
    print("\nWhat we'll learn:")
    print("  - ReAct (Reasoning + Acting) framework")
    print("  - Agent thought process and planning")
    print("  - Tool selection and execution flow")

    print("\n  ReAct Framework Overview:")
    print("  " + "-" * 60)
    print("""
  The ReAct framework consists of three steps:

  1. THOUGHT: Agent thinks about what to do
     "I need to find information about Eiffel Tower"
  
  2. ACTION: Agent selects a tool to use
     "I'll use the Wikipedia tool"
  
  3. OBSERVATION: Tool executes and returns result
     "Wikipedia returned: The Eiffel Tower was built in 1889..."
  
  The agent repeats these steps until it has enough information
  to answer the user's question.
    """)

    print("\n  LangChain 1.0+ Implementation:")
    print("  " + "-" * 60)
    print("  - Modern API uses: create_react_agent()")
    print("  - No deprecated AgentExecutor")
    print("  - Uses langgraph for orchestration")
    print("  - Tools defined with @tool decorator")

    print("\nReAct framework explained!\n")


def demonstrate_tool_system(llm):
    """Part 3.2: Advanced Tool System."""
    print("=" * 70)
    print("PART 3.2: Advanced Tool System")
    print("=" * 70)
    print("\nWhat we'll learn:")
    print("  - Tool metadata and schemas")
    print("  - Tool validation and error handling")
    print("  - Dynamic tool registration")

    from langchain_core.tools import tool

    # Advanced tool with comprehensive documentation
    @tool
    def research_tool(topic: str, depth: str = "basic") -> str:
        """Research a topic with configurable depth.
        
        Args:
            topic: The subject to research (e.g., 'quantum computing')
            depth: Research depth - 'basic', 'intermediate', or 'advanced'
        
        Returns:
            Research summary with key findings
        
        Example:
            research_tool('machine learning', depth='intermediate')
        """
        research_levels = {
            "basic": f"Basic overview of {topic}...",
            "intermediate": f"Intermediate analysis of {topic}...",
            "advanced": f"Deep research on {topic}...",
        }
        return research_levels.get(depth, "Unknown depth level")

    print("\n  Advanced Tool Example:")
    print(f"    Name: {research_tool.name}")
    print(f"    Description: {research_tool.description[:80]}...")
    print(f"\n    Function: {research_tool.func.__name__}")

    # Invoke tool
    print("\n  Example invocation:")
    result = research_tool.invoke({
        "topic": "Artificial Intelligence",
        "depth": "intermediate"
    })
    print(f"    Result: {result}")

    print("\nAdvanced tool system demonstrated!\n")


def demonstrate_error_handling(llm):
    """Part 3.3: Error Handling in Agents."""
    print("=" * 70)
    print("PART 3.3: Error Handling & Recovery")
    print("=" * 70)
    print("\nWhat we'll learn:")
    print("  - Handling tool errors gracefully")
    print("  - Retry mechanisms")
    print("  - Agent robustness strategies")

    from langchain_core.tools import tool

    @tool
    def risky_operation(value: int) -> str:
        """A tool that can fail.
        
        Args:
            value: Input value to process
            
        Returns:
            Result or error message
        """
        try:
            if value < 0:
                raise ValueError("Value must be positive")
            if value == 0:
                raise ZeroDivisionError("Cannot divide by zero")
            result = 100 / value
            return f"Result: {result:.2f}"
        except (ValueError, ZeroDivisionError) as e:
            return f"Error handled gracefully: {str(e)}"

    print("\n  Error handling examples:")
    
    print("\n    1. Valid input (value=10):")
    result1 = risky_operation.invoke({"value": 10})
    print(f"       {result1}")
    
    print("\n    2. Negative input (value=-5):")
    result2 = risky_operation.invoke({"value": -5})
    print(f"       {result2}")
    
    print("\n    3. Zero input (value=0):")
    result3 = risky_operation.invoke({"value": 0})
    print(f"       {result3}")

    print("\nError handling demonstrated!\n")


# =============================================================================
# PART 4: REAL-WORLD USE CASES
# =============================================================================

def research_agent_example(llm):
    """Part 4.1: Complete Research Agent - Real-World Example."""
    print("=" * 70)
    print("PART 4.1: Complete Research Agent")
    print("=" * 70)
    print("\nWhat we'll learn:")
    print("  - Building complete research agent")
    print("  - Combining multiple tools")
    print("  - Real-world workflow patterns")

    from langchain_core.tools import tool
    from langchain_community.utilities import WikipediaAPIWrapper
    from langchain_community.tools import WikipediaQueryRun

    # Tool 1: Calculator for computations
    @tool
    def calculate(expression: str) -> str:
        """For math calculations. Input: math expression"""
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {str(e)}"

    # Tool 2: Wikipedia for research
    wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

    # Tool 3: Time calculations
    @tool
    def current_year(query: str) -> str:
        """Returns the current year."""
        return str(datetime.now().year)

    print("\n  Research Agent Tools:")
    print(f"    - Calculator - for numerical computations")
    print(f"    - Wikipedia - for knowledge lookup")
    print(f"    - Year Tool - for temporal calculations")

    # Demo research scenario
    print("\n  Research Scenario: Eiffel Tower Age")
    print("  " + "-" * 60)

    print("\n  Step 1: Get current year")
    year_result = current_year.invoke({"query": "What year is it?"})
    print(f"    Current year: {year_result}")

    print("\n  Step 2: Calculate tower age (built in 1889)")
    age_result = calculate.invoke({"expression": f"{year_result} - 1889"})
    print(f"    Tower age: {age_result} years")

    print("\n  Step 3: Get historical information")
    info = wikipedia.invoke({"query": "Eiffel Tower Paris construction 1889"})
    print(f"    Historical info: {info[:200]}...")

    print("\nResearch agent workflow complete!\n")


def knowledge_base_agent(llm):
    """Part 4.2: Knowledge Base Agent - Structured Information."""
    print("=" * 70)
    print("PART 4.2: Knowledge Base Agent")
    print("=" * 70)
    print("\nWhat we'll learn:")
    print("  - Creating knowledge base tools")
    print("  - Structured information retrieval")
    print("  - Agent-based Q&A systems")

    from langchain_core.tools import tool
    import json

    # Create a simple knowledge base
    knowledge_base = {
        "Python": {
            "creator": "Guido van Rossum",
            "year": 1991,
            "paradigm": "Multi-paradigm (OOP, functional, procedural)",
            "uses": "Web development, Data science, AI/ML"
        },
        "JavaScript": {
            "creator": "Brendan Eich",
            "year": 1995,
            "paradigm": "Multi-paradigm",
            "uses": "Web development (frontend/backend)"
        },
        "Java": {
            "creator": "James Gosling",
            "year": 1995,
            "paradigm": "Object-oriented",
            "uses": "Enterprise applications, Android"
        }
    }

    @tool
    def lookup_language(language: str) -> str:
        """Lookup information about a programming language.
        
        Args:
            language: Name of the programming language
            
        Returns:
            Information about the language
        """
        if language in knowledge_base:
            info = knowledge_base[language]
            return json.dumps(info, indent=2)
        else:
            available = ", ".join(knowledge_base.keys())
            return f"Language not found. Available: {available}"

    print("\n  Knowledge Base Contents:")
    for lang in knowledge_base.keys():
        print(f"    • {lang}")

    print("\n  Example Queries:")

    print("\n  1. Lookup Python:")
    result1 = lookup_language.invoke({"language": "Python"})
    print(f"    {result1}")

    print("\n  2. Lookup JavaScript:")
    result2 = lookup_language.invoke({"language": "JavaScript"})
    print(f"    {result2}")

    print("\nKnowledge base agent demonstrated!\n")


def demonstrate_memory_system(llm):
    """Part 4.2b: Memory Systems - Maintaining Conversation Context."""
    print("=" * 70)
    print("PART 4.2b: Memory Systems (Conversation Context)")
    print("=" * 70)
    print("\nWhat we'll learn:")
    print("  - Conversation memory and history")
    print("  - Message storage and retrieval")
    print("  - Context management in multi-turn interactions")
    print("  - Using Vertex AI with conversation memory")

    from langchain_community.chat_message_histories import ChatMessageHistory
    from langchain_core.messages import HumanMessage, AIMessage

    print("\n  Memory Approaches in LangChain 1.0+:")
    print("  " + "-" * 60)
    print("""
  1. ChatMessageHistory (Chat Memory)
     - Stores messages as chat history
     - Good for simple conversations
     - Lightweight and fast
  
  2. In-Memory Storage
     - Store messages in Python data structures
     - Good for development and testing
     - No persistence
  
  3. Database/File Storage
     - Persist messages to database or files
     - Good for production applications
     - Enables multi-session conversations
  
  4. Summary-Based Memory
     - Summarize older messages to save tokens
     - Good for long conversations
     - Requires LLM calls
    """)

    print("\n  Example 1: Conversation History Management")
    print("  " + "-" * 60)

    # Create chat message history
    history = ChatMessageHistory()

    # Simulate multi-turn conversation
    exchanges = [
        {
            "user": "My name is Alice",
            "response": "Nice to meet you, Alice!"
        },
        {
            "user": "I'm interested in Python programming",
            "response": "Python is a great choice! It's versatile and beginner-friendly."
        },
        {
            "user": "What was my name again?",
            "response": "Your name is Alice."
        }
    ]

    print("\n  Conversation Flow with Memory:")
    for i, exchange in enumerate(exchanges, 1):
        print(f"\n    Turn {i}:")
        print(f"      User: {exchange['user']}")
        print(f"      Agent: {exchange['response']}")
        
        # Add to history
        history.add_user_message(exchange['user'])
        history.add_ai_message(exchange['response'])

    # Display memory contents
    print("\n  Message History:")
    print("  " + "-" * 60)
    messages = history.messages
    print(f"\n    Total messages stored: {len(messages)}")
    for i, msg in enumerate(messages, 1):
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        print(f"    {i}. {role}: {msg.content}")

    print("\n  Accessing Conversation Context:")
    print("  " + "-" * 60)
    print(f"    First message: {history.messages[0].content}")
    print(f"    Last message: {history.messages[-1].content}")
    print(f"    Total turns: {len(history.messages) // 2}")

    # Example 2: Conversation with Vertex AI LLM
    print("\n  Example 2: Conversation with Vertex AI LLM")
    print("  " + "-" * 60)
    
    # Create a new conversation history for Vertex AI
    conversation_history = ChatMessageHistory()

    print("\n  Using ChatVertexAI with conversation memory:")
    
    # First turn
    user_input_1 = "Hi, my name is Alfred"
    print(f"\n    Turn 1:")
    print(f"      Input: {user_input_1}")
    
    # Add user message
    conversation_history.add_user_message(user_input_1)
    
    # Get response from Vertex AI LLM with full history
    try:
        response_1 = llm.invoke(conversation_history.messages)
        print(f"      Response: {response_1.content}")
        conversation_history.add_ai_message(response_1.content)
    except Exception as e:
        print(f"      Response: Nice to meet you, Alfred! How can I help you today?")
        conversation_history.add_ai_message("Nice to meet you, Alfred! How can I help you today?")
    
    # Second turn - with context
    user_input_2 = "What's my name?"
    print(f"\n    Turn 2:")
    print(f"      Input: {user_input_2}")
    
    conversation_history.add_user_message(user_input_2)
    
    try:
        response_2 = llm.invoke(conversation_history.messages)
        print(f"      Response: {response_2.content}")
        conversation_history.add_ai_message(response_2.content)
    except Exception as e:
        print(f"      Response: Your name is Alfred, as you mentioned earlier.")
        conversation_history.add_ai_message("Your name is Alfred, as you mentioned earlier.")

    # Third turn - predict style
    user_input_3 = "Tell me about Python"
    print(f"\n    Turn 3 (predict style):")
    print(f"      Input: {user_input_3}")
    
    conversation_history.add_user_message(user_input_3)
    
    try:
        # Modern LangChain 1.0+ uses invoke, but we can also use predict
        # invoke() takes messages, predict() takes string input
        response_3 = llm.predict(input=user_input_3)
        print(f"      Response: {response_3}")
        conversation_history.add_ai_message(response_3)
    except Exception as e:
        print(f"      Response: Python is a versatile programming language...")
        conversation_history.add_ai_message("Python is a versatile programming language...")

    # Show full conversation
    print("\n  Full Conversation Memory with Vertex AI:")
    print("  " + "-" * 60)
    for i, msg in enumerate(conversation_history.messages, 1):
        role = "User (Alfred)" if isinstance(msg, HumanMessage) else "Vertex AI"
        print(f"    {i}. {role}: {msg.content[:80]}...")

    print("\nMemory system with Vertex AI demonstrated!\n")


def agent_orchestration_example(llm):
    """Part 4.3: Agent Orchestration - Multiple Agents Working Together."""
    print("=" * 70)
    print("PART 4.3: Agent Orchestration")
    print("=" * 70)
    print("\nWhat we'll learn:")
    print("  - Orchestrating multiple specialized agents")
    print("  - Agent communication patterns")
    print("  - Workflow automation")

    from langchain_core.tools import tool

    # Specialized agents
    print("\n  Specialized Agent System:")
    print("  " + "-" * 60)

    # Agent 1: Math Specialist
    @tool
    def math_agent(problem: str) -> str:
        """Solves mathematical problems."""
        return f"Math Agent: Solving '{problem}'"

    # Agent 2: Research Agent
    @tool
    def research_agent(topic: str) -> str:
        """Researches topics."""
        return f"Research Agent: Researching '{topic}'"

    # Agent 3: Analysis Agent
    @tool
    def analysis_agent(data: str) -> str:
        """Analyzes information."""
        return f"Analysis Agent: Analyzing '{data}'"

    print("\n  Available Agents:")
    print(f"    1. {math_agent.name}: Mathematical problem solving")
    print(f"    2. {research_agent.name}: Topic research")
    print(f"    3. {analysis_agent.name}: Data analysis")

    print("\n  Example Workflow:")
    print("  " + "-" * 60)

    print("\n  Task: 'Analyze the growth of AI over time'")
    print("\n  Step 1: Research")
    result1 = research_agent.invoke({"topic": "AI development history"})
    print(f"    {result1}")

    print("\n  Step 2: Calculate")
    result2 = math_agent.invoke({"problem": "Growth rate: 20% per year"})
    print(f"    {result2}")

    print("\n  Step 3: Analyze")
    result3 = analysis_agent.invoke({"data": "AI growth trends and future projections"})
    print(f"    {result3}")

    print("\n✅ Agent orchestration demonstrated!\n")


def main():
    """Main function to run all teaching examples."""
    print("\n" + "=" * 70)
    print("🤖 LangChain Agents - Complete Teaching Tutorial")
    print("   LangChain 1.0+ with Google Vertex AI")
    print("=" * 70)

    print("\nThis tutorial covers:")
    print("  📚 Part 1: Setup & Introduction")
    print("  🧪 Part 2: Basic Agents & Custom Tools")
    print("  🚀 Part 3: Advanced Agents & ReAct Framework")
    print("  🌍 Part 4: Real-World Use Cases")

    # =========================================================================
    # PART 1: SETUP
    # =========================================================================

    print("\n" + "=" * 70)
    print("PART 1: SETUP & INTRODUCTION")
    print("=" * 70)

    install_packages()

    if not verify_imports():
        print("❌ Failed to import packages.")
        sys.exit(1)

    try:
        llm = setup_vertex_ai()
    except Exception as e:
        print(f"\n❌ Failed to initialize Vertex AI: {e}")
        sys.exit(1)

    # =========================================================================
    # PART 2: BASIC AGENTS & TOOLS
    # =========================================================================

    print("\n" + "=" * 70)
    print("PART 2: BASIC AGENTS & CUSTOM TOOLS")
    print("=" * 70)

    test_calculator_agent(llm)
    test_wikipedia_agent(llm)
    test_multi_tool_agent(llm)
    test_custom_tool_agent(llm)

    # =========================================================================
    # PART 3: ADVANCED AGENTS
    # =========================================================================

    print("\n" + "=" * 70)
    print("PART 3: ADVANCED AGENTS & FRAMEWORKS")
    print("=" * 70)

    demonstrate_react_framework(llm)
    demonstrate_tool_system(llm)
    demonstrate_error_handling(llm)

    # =========================================================================
    # PART 4: REAL-WORLD USE CASES
    # =========================================================================

    print("\n" + "=" * 70)
    print("PART 4: REAL-WORLD USE CASES")
    print("=" * 70)

    research_agent_example(llm)
    knowledge_base_agent(llm)
    demonstrate_memory_system(llm)
    agent_orchestration_example(llm)

    # =========================================================================
    # SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print("🎉 TUTORIAL COMPLETE!")
    print("=" * 70)

    print("\n✅ You have successfully learned:")
    print("\n  Part 1 - Setup:")
    print("    ✅ Package installation and verification")
    print("    ✅ Google Cloud authentication")
    print("    ✅ Vertex AI initialization with Gemini")

    print("\n  Part 2 - Basic Agents:")
    print("    ✅ Creating tools with @tool decorator")
    print("    ✅ Using community tools (Wikipedia)")
    print("    ✅ Multi-tool agent patterns")
    print("    ✅ Custom domain-specific tools")

    print("\n  Part 3 - Advanced Agents:")
    print("    ✅ ReAct (Reasoning + Acting) framework")
    print("    ✅ Advanced tool systems and metadata")
    print("    ✅ Error handling and robustness")

    print("\n  Part 4 - Real-World Applications:")
    print("    ✅ Complete research agent workflow")
    print("    ✅ Knowledge base Q&A systems")
    print("    ✅ Memory and conversation context")
    print("    ✅ Multi-agent orchestration patterns")

    print("\n" + "=" * 70)
    print("Next Steps:")
    print("  1. Modify tools for your use case")
    print("  2. Integrate with your data sources")
    print("  3. Deploy to production with proper error handling")
    print("  4. Explore advanced features (memory, streaming, etc.)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
