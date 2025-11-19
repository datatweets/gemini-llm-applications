#!/usr/bin/env python3
"""
Complete Conversation Memory Tutorial with Vertex AI - LangChain 1.0+
======================================================================

This comprehensive tutorial teaches all conversation memory concepts:

Part 1 - Basic Memory:
  1. Simple conversation with ChatMessageHistory
  2. Building a predict() function (replaces deprecated ConversationChain)
  3. Multi-turn conversations with full context

Part 2 - Advanced Memory Types:
  4. ConversationBufferMemory - Store all messages
  5. ConversationBufferWindowMemory - Keep last K exchanges only
  6. ConversationTokenBufferMemory - Keep messages within token limit
  7. ConversationSummaryBufferMemory - Summarize old messages

Prerequisites:
- Python 3.10+
- gcloud auth application-default login
- GCP project with Vertex AI enabled
- LangChain 1.0+

Installation:
    pip install langchain langchain-google-vertexai langchain-community

Usage:
    python conversation_memory_complete_tutorial.py
"""

import vertexai
import time
from langchain_google_vertexai import ChatVertexAI
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ID = "terraform-prj-476214"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.0-flash-exp"


# =============================================================================
# SETUP: Initialize Vertex AI
# =============================================================================

def setup_vertex_ai():
    """
    Initialize Vertex AI and create LLM.
    
    Returns:
        ChatVertexAI: Configured LLM instance
    """
    print("\n" + "=" * 70)
    print("Initializing Vertex AI...")
    print("=" * 70)
    
    # Initialize Vertex AI with project and location
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    # Create LLM for text generation
    llm = ChatVertexAI(
        model=MODEL_NAME,
        project=PROJECT_ID,
        location=LOCATION,
        temperature=0.7,  # Controls randomness (0=deterministic, 1=creative)
        max_tokens=1024,  # Maximum response length
    )
    
    print(f"✅ LLM initialized: {MODEL_NAME}")
    print(f"   Project: {PROJECT_ID}")
    print(f"   Location: {LOCATION}")
    print("=" * 70)
    
    return llm


# =============================================================================
# PART 1: BASIC CONVERSATION MEMORY
# =============================================================================

# -----------------------------------------------------------------------------
# Section 1: Simple Conversation with ChatMessageHistory
# -----------------------------------------------------------------------------

def demo_basic_memory(llm):
    """
    Demonstrates basic conversation memory using ChatMessageHistory.
    
    ChatMessageHistory stores messages in memory (not persistent).
    Each message is either HumanMessage or AIMessage.
    
    Args:
        llm: ChatVertexAI instance
    """
    print("\n" + "=" * 70)
    print("SECTION 1: Basic Conversation Memory")
    print("=" * 70)
    print("\nChatMessageHistory stores conversation messages in memory.")
    print("This allows the AI to remember previous exchanges.\n")
    
    # Create message history storage
    history = ChatMessageHistory()
    
    # Turn 1: Introduction
    print("Turn 1: User introduces themselves")
    user_msg_1 = "Hi, my name is Alice and I'm learning LangChain"
    print(f"  User: {user_msg_1}")
    
    # Add user message to history
    history.add_user_message(user_msg_1)
    
    # Get AI response with current history
    time.sleep(2)  # Proactive rate limiting
    response_1 = llm.invoke(history.messages)
    print(f"  AI: {response_1.content}")
    
    # Add AI response to history
    history.add_ai_message(response_1.content)
    
    # Turn 2: Test memory
    print("\nTurn 2: Test if AI remembers the name")
    user_msg_2 = "What is my name?"
    print(f"  User: {user_msg_2}")
    
    history.add_user_message(user_msg_2)
    time.sleep(2)  # Proactive rate limiting
    response_2 = llm.invoke(history.messages)
    print(f"  AI: {response_2.content}")
    
    history.add_ai_message(response_2.content)
    
    # Turn 3: Test context retention
    print("\nTurn 3: Test if AI remembers what I'm learning")
    user_msg_3 = "What am I learning?"
    print(f"  User: {user_msg_3}")
    
    history.add_user_message(user_msg_3)
    time.sleep(2)  # Proactive rate limiting
    response_3 = llm.invoke(history.messages)
    print(f"  AI: {response_3.content}")
    
    history.add_ai_message(response_3.content)
    
    # Display full conversation history
    print("\n" + "-" * 70)
    print(f"Full Conversation History ({len(history.messages)} messages):")
    print("-" * 70)
    for i, msg in enumerate(history.messages, 1):
        role = "👤 User" if isinstance(msg, HumanMessage) else "🤖 AI"
        print(f"{i}. {role}: {msg.content}")
    
    print("\n💡 Key Takeaway: ChatMessageHistory maintains context across turns!")
    print("=" * 70)


# -----------------------------------------------------------------------------
# Section 2: Building a predict() Function
# -----------------------------------------------------------------------------

def create_conversation_with_memory(llm):
    """
    Creates a conversation function that mimics the old ConversationChain.predict().
    
    In LangChain 1.0+, ConversationChain is deprecated.
    This shows how to build equivalent functionality.
    
    Args:
        llm: ChatVertexAI instance
        
    Returns:
        tuple: (predict_function, history) for managing conversation
    """
    # Create message history
    history = ChatMessageHistory()
    
    def predict(input_text):
        """
        Predict response with conversation memory.
        
        This mimics the old ConversationChain.predict() method.
        
        Args:
            input_text: User's input message
            
        Returns:
            str: AI's response
        """
        # Add user message to history
        history.add_user_message(input_text)
        
        # Get AI response with full conversation context
        time.sleep(2)  # Proactive rate limiting
        response = llm.invoke(history.messages)
        
        # Add AI response to history
        history.add_ai_message(response.content)
        
        return response.content
    
    return predict, history


def demo_predict_function(llm):
    """
    Demonstrates the predict() function for easier conversation management.
    
    Args:
        llm: ChatVertexAI instance
    """
    print("\n" + "=" * 70)
    print("SECTION 2: Building a predict() Function")
    print("=" * 70)
    print("\nThe predict() function simplifies conversation management.")
    print("It automatically handles message history.\n")
    
    # Create conversation with memory
    predict, history = create_conversation_with_memory(llm)
    
    # Conduct multi-turn conversation
    turns = [
        ("User introduces hobby", "I love playing guitar and composing music"),
        ("AI should remember hobby", "What are my hobbies?"),
        ("New topic", "What is 5 + 7?"),
        ("Return to earlier topic", "What instrument do I play?"),
    ]
    
    for i, (description, user_input) in enumerate(turns, 1):
        print(f"\nTurn {i}: {description}")
        print(f"  Input: {user_input}")
        
        # Simply call predict() - it handles everything!
        response = predict(user_input)
        print(f"  Output: {response}")
    
    # Show conversation history
    print("\n" + "-" * 70)
    print(f"Conversation History ({len(history.messages)} messages):")
    print("-" * 70)
    for msg in history.messages:
        role = "User" if isinstance(msg, HumanMessage) else "AI"
        print(f"{role}: {msg.content}")
    
    print("\n💡 Key Takeaway: predict() encapsulates memory management!")
    print("=" * 70)


# =============================================================================
# PART 2: ADVANCED MEMORY TYPES
# =============================================================================

class ConversationWithMemory:
    """
    Advanced conversation memory implementation.
    
    Supports multiple memory types:
    - buffer: Keep all messages (default)
    - window: Keep last K exchanges
    - token: Keep messages within token limit
    - summary: Summarize old messages, keep recent ones
    """
    
    def __init__(self, llm, memory_type="buffer", **memory_kwargs):
        """
        Initialize conversation with specified memory type.
        
        Args:
            llm: ChatVertexAI instance
            memory_type: Type of memory ('buffer', 'window', 'token', 'summary')
            **memory_kwargs: Additional memory configuration
                - k: Number of exchanges to keep (for window)
                - max_token_limit: Token limit (for token/summary)
        """
        self.llm = llm
        self.history = ChatMessageHistory()
        self.memory_type = memory_type
        self.memory_kwargs = memory_kwargs
        
        # Memory type specific parameters
        self.k = memory_kwargs.get('k', 5)
        self.max_token_limit = memory_kwargs.get('max_token_limit', 2000)
        self.summary_token_limit = memory_kwargs.get('max_token_limit', 100)
    
    def _count_tokens(self, text):
        """Rough token count estimation (1 token ≈ 4 chars)."""
        return len(text) // 4
    
    def _get_working_messages(self):
        """
        Get messages based on memory type.
        
        Returns:
            list: Messages to use for LLM context
        """
        all_messages = self.history.messages
        
        if self.memory_type == "window":
            # Keep only last K exchanges (2K messages for K exchanges)
            return all_messages[-(self.k * 2):]
        
        elif self.memory_type == "token":
            # Keep messages within token limit
            messages = []
            total_tokens = 0
            
            # Add messages from most recent backwards
            for msg in reversed(all_messages):
                msg_tokens = self._count_tokens(msg.content)
                if total_tokens + msg_tokens <= self.max_token_limit:
                    messages.insert(0, msg)
                    total_tokens += msg_tokens
                else:
                    break
            return messages
        
        elif self.memory_type == "summary":
            # Use summary + recent messages
            recent_messages = all_messages[-4:]  # Keep last 2 exchanges
            
            # Calculate tokens in recent messages
            recent_tokens = sum(self._count_tokens(m.content) for m in recent_messages)
            
            # If we exceed limit, create summary
            if recent_tokens > self.summary_token_limit and len(all_messages) > 4:
                # Summarize older messages
                old_messages = all_messages[:-4]
                summary_text = self._create_summary(old_messages)
                
                # Return summary as system message + recent messages
                return [SystemMessage(content=f"Previous conversation summary: {summary_text}")] + recent_messages
            
            return all_messages
        
        else:  # buffer - return all messages
            return all_messages
    
    def _create_summary(self, messages):
        """
        Create a summary of older messages.
        
        Args:
            messages: Messages to summarize
            
        Returns:
            str: Summary text
        """
        if not messages:
            return ""
        
        # Create prompt for summarization
        conversation_text = "\n".join([
            f"{'Human' if isinstance(m, HumanMessage) else 'AI'}: {m.content}"
            for m in messages
        ])
        
        summary_prompt = f"Summarize this conversation briefly:\n{conversation_text}"
        
        try:
            time.sleep(2)  # Proactive rate limiting
            summary_response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            return summary_response.content
        except Exception:
            return "Previous conversation occurred."
    
    def predict(self, input_text):
        """
        Predict response with memory management.
        
        Args:
            input_text: User input
            
        Returns:
            str: AI response
        """
        # Add user message to full history
        self.history.add_user_message(input_text)
        
        # Get working messages based on memory type
        working_messages = self._get_working_messages()
        
        # Proactive rate limiting to avoid quota errors
        time.sleep(3)
        
        # Get response from LLM
        response = self.llm.invoke(working_messages)
        
        # Add AI response to history
        self.history.add_ai_message(response.content)
        
        return response.content
    
    def get_history(self):
        """Return all messages in conversation history."""
        return self.history.messages
    
    def get_working_memory(self):
        """Return messages currently in working memory."""
        return self._get_working_messages()
    
    def save_context(self, input_dict, output_dict):
        """
        Save context to memory (mimics old memory.save_context).
        
        Args:
            input_dict: Dict with 'input' key
            output_dict: Dict with 'output' key
        """
        self.history.add_user_message(input_dict["input"])
        self.history.add_ai_message(output_dict["output"])


# -----------------------------------------------------------------------------
# Section 3: ConversationBufferMemory (Store All Messages)
# -----------------------------------------------------------------------------

def demo_buffer_memory(llm):
    """
    Demonstrates buffer memory - stores all messages.
    
    Use when:
    - Conversation is short
    - You need full context
    - Token limits are not a concern
    
    Args:
        llm: ChatVertexAI instance
    """
    print("\n" + "=" * 70)
    print("SECTION 3: ConversationBufferMemory")
    print("=" * 70)
    print("\nStores ALL messages in memory.")
    print("Provides complete conversation context.\n")
    
    # Create conversation with buffer memory
    conversation = ConversationWithMemory(llm, memory_type="buffer")
    
    # Multi-turn conversation
    turns = [
        "Hi, I'm Bob and I work as a teacher",
        "What is my profession?",
        "I teach mathematics at a high school",
        "What subject do I teach and where?",
    ]
    
    for i, user_input in enumerate(turns, 1):
        print(f"Turn {i}:")
        print(f"  User: {user_input}")
        
        response = conversation.predict(user_input)
        print(f"  AI: {response}\n")
    
    print(f"Total messages in history: {len(conversation.get_history())}")
    print("Working memory: ALL messages retained")
    
    print("\n💡 Buffer memory is simple but can grow large for long conversations")
    print("=" * 70)


# -----------------------------------------------------------------------------
# Section 4: ConversationBufferWindowMemory (Keep Last K Exchanges)
# -----------------------------------------------------------------------------

def demo_window_memory(llm):
    """
    Demonstrates window memory - keeps only last K conversation turns.
    
    Use when:
    - Only recent context matters
    - Want to limit memory usage
    - Conversation is long
    
    Args:
        llm: ChatVertexAI instance
    """
    print("\n" + "=" * 70)
    print("SECTION 4: ConversationBufferWindowMemory (k=1)")
    print("=" * 70)
    print("\nKeeps only the last K exchanges in memory.")
    print("With k=1, only the most recent exchange is remembered.\n")
    
    # Create conversation with window memory (k=1)
    conversation = ConversationWithMemory(llm, memory_type="window", k=1)
    
    # Turn 1
    print("Turn 1:")
    print("  Input: 'Hi, my name is Charlie'")
    response1 = conversation.predict("Hi, my name is Charlie")
    print(f"  Output: {response1}\n")
    
    # Turn 2 - should remember name (within window)
    print("Turn 2:")
    print("  Input: 'What is my name?'")
    response2 = conversation.predict("What is my name?")
    print(f"  Output: {response2}")
    print("  ✅ Name remembered (within k=1 window)\n")
    
    # Turn 3 - new topic
    print("Turn 3:")
    print("  Input: 'What is 10 * 10?'")
    response3 = conversation.predict("What is 10 * 10?")
    print(f"  Output: {response3}")
    print("  (Turn 1 pushed out of window)\n")
    
    # Turn 4 - should NOT remember name (outside window)
    print("Turn 4:")
    print("  Input: 'What is my name?'")
    response4 = conversation.predict("What is my name?")
    print(f"  Output: {response4}")
    print("  ❌ Name forgotten (outside k=1 window)")
    
    print(f"\nTotal messages in full history: {len(conversation.get_history())}")
    print(f"Messages in working memory: {len(conversation.get_working_memory())}")
    
    print("\n💡 Window memory is great for focusing on recent context")
    print("=" * 70)


# -----------------------------------------------------------------------------
# Section 5: ConversationTokenBufferMemory (Token Limit)
# -----------------------------------------------------------------------------

def demo_token_buffer_memory(llm):
    """
    Demonstrates token buffer memory - keeps messages within token limit.
    
    Use when:
    - Have strict token limits
    - Want automatic pruning of old messages
    - Conversation length varies
    
    Args:
        llm: ChatVertexAI instance
    """
    print("\n" + "=" * 70)
    print("SECTION 5: ConversationTokenBufferMemory (max_token_limit=50)")
    print("=" * 70)
    print("\nKeeps messages within a token limit.")
    print("Old messages are dropped when limit is exceeded.\n")
    
    # Create conversation with token buffer memory
    conversation = ConversationWithMemory(
        llm, 
        memory_type="token", 
        max_token_limit=50  # Very small limit for demonstration
    )
    
    # Turn 1
    print("Turn 1:")
    print("  Input: 'Hi, my name is Diana'")
    response1 = conversation.predict("Hi, my name is Diana")
    print(f"  Output: {response1}\n")
    
    # Turn 2
    print("Turn 2:")
    print("  Input: 'What is my name?'")
    response2 = conversation.predict("What is my name?")
    print(f"  Output: {response2}\n")
    
    # Turn 3
    print("Turn 3:")
    print("  Input: 'What is the capital of France?'")
    response3 = conversation.predict("What is the capital of France?")
    print(f"  Output: {response3}\n")
    
    # Turn 4 - might not remember due to token limit
    print("Turn 4:")
    print("  Input: 'What is my name?'")
    response4 = conversation.predict("What is my name?")
    print(f"  Output: {response4}")
    
    print(f"\nTotal messages in full history: {len(conversation.get_history())}")
    print(f"Messages in working memory: {len(conversation.get_working_memory())}")
    print("  (Older messages dropped to stay within token limit)")
    
    print("\n💡 Token buffer ensures you never exceed LLM token limits")
    print("=" * 70)


# -----------------------------------------------------------------------------
# Section 6: ConversationSummaryBufferMemory (Summarize Old Messages)
# -----------------------------------------------------------------------------

def demo_summary_buffer_memory(llm):
    """
    Demonstrates summary buffer memory - summarizes old messages.
    
    Use when:
    - Need full conversation context but have token limits
    - Older details can be summarized
    - Want best of both worlds
    
    Args:
        llm: ChatVertexAI instance
    """
    print("\n" + "=" * 70)
    print("SECTION 6: ConversationSummaryBufferMemory (max_token_limit=100)")
    print("=" * 70)
    print("\nKeeps recent messages in full, summarizes older ones.")
    print("Ideal for long conversations with token limits.\n")
    
    # Create conversation with summary buffer memory
    memory = ConversationWithMemory(
        llm,
        memory_type="summary",
        max_token_limit=100
    )
    
    # Save a detailed schedule using save_context
    schedule = """There is a meeting at 8am with your product team. 
You will need your powerpoint presentation prepared. 
9am-12pm have time to work on your LangChain project which will go quickly 
because LangChain is such a powerful tool. 
At Noon, lunch at the italian restaurant with a customer who is driving 
from over an hour away to meet you to understand the latest in AI. 
Be sure to bring your laptop to show the latest LLM demo."""
    
    print("Saving conversation context:")
    print("1. 'Hello' -> 'What's up'")
    memory.save_context({"input": "Hello"}, {"output": "What's up"})
    
    print("2. 'Not much, just hanging' -> 'Cool'")
    memory.save_context(
        {"input": "Not much, just hanging"},
        {"output": "Cool"}
    )
    
    print("3. 'What is on the schedule today?' -> [Detailed schedule]")
    memory.save_context(
        {"input": "What is on the schedule today?"},
        {"output": schedule}
    )
    
    print("\n" + "-" * 70)
    print(f"Total messages in history: {len(memory.get_history())}")
    print(f"Messages in working memory: {len(memory.get_working_memory())}")
    print("-" * 70)
    
    # Ask a question based on the schedule
    print("\nAsking a question:")
    print("  Input: 'What would be a good demo to show?'")
    response = memory.predict("What would be a good demo to show?")
    print(f"  Output: {response}")
    
    print("\n💡 Summary memory provides context while managing token limits")
    print("=" * 70)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Execute all tutorial sections in sequence."""
    print("\n" + "=" * 70)
    print("Complete Conversation Memory Tutorial")
    print("=" * 70)
    print("\n📚 This tutorial covers:")
    print("\n   Part 1 - Basic Memory:")
    print("     1. ChatMessageHistory basics")
    print("     2. Building predict() function")
    print("\n   Part 2 - Advanced Memory Types:")
    print("     3. Buffer Memory (store all)")
    print("     4. Window Memory (last K exchanges)")
    print("     5. Token Buffer Memory (token limit)")
    print("     6. Summary Buffer Memory (summarize old)")
    print("\n⏱️  Rate limiting enabled to avoid API quota errors")
    print("=" * 70)
    
    # Initialize Vertex AI
    llm = setup_vertex_ai()
    
    # PART 1: BASIC MEMORY
    print("\n\n" + "=" * 70)
    print("PART 1: BASIC CONVERSATION MEMORY")
    print("=" * 70)
    
    demo_basic_memory(llm)
    
    print("\n⏳ Waiting 3 seconds before next section...\n")
    time.sleep(3)
    
    demo_predict_function(llm)
    
    # PART 2: ADVANCED MEMORY TYPES
    print("\n\n" + "=" * 70)
    print("PART 2: ADVANCED MEMORY TYPES")
    print("=" * 70)
    
    print("\n⏳ Waiting 3 seconds before next section...\n")
    time.sleep(3)
    
    demo_buffer_memory(llm)
    
    print("\n⏳ Waiting 3 seconds before next section...\n")
    time.sleep(3)
    
    demo_window_memory(llm)
    
    print("\n⏳ Waiting 3 seconds before next section...\n")
    time.sleep(3)
    
    demo_token_buffer_memory(llm)
    
    print("\n⏳ Waiting 3 seconds before next section...\n")
    time.sleep(3)
    
    demo_summary_buffer_memory(llm)
    
    # Final summary
    print("\n\n" + "=" * 70)
    print("✅ Complete Tutorial Finished!")
    print("=" * 70)
    print("\n📋 What You Learned:")
    print("\n   Basic Memory:")
    print("     ✓ ChatMessageHistory for storing conversations")
    print("     ✓ Building predict() function (replaces ConversationChain)")
    print("     ✓ Managing multi-turn conversations")
    print("\n   Advanced Memory Types:")
    print("     ✓ Buffer: Store all messages")
    print("     ✓ Window: Keep last K exchanges")
    print("     ✓ Token Buffer: Stay within token limits")
    print("     ✓ Summary Buffer: Summarize old, keep recent")
    print("\n🎯 Key Concepts:")
    print("   • Memory enables context-aware conversations")
    print("   • Different memory types suit different use cases")
    print("   • LangChain 1.0+ uses modern APIs (no deprecated chains)")
    print("   • Choose memory type based on conversation length and limits")
    print("\n💡 When to Use Each Memory Type:")
    print("   • Buffer: Short conversations, need full context")
    print("   • Window: Only recent context matters")
    print("   • Token Buffer: Strict token limits")
    print("   • Summary Buffer: Long conversations with token limits")
    print("\n🚀 Next Steps:")
    print("   • Experiment with different memory types")
    print("   • Adjust k values and token limits")
    print("   • Build your own conversational applications")
    print("   • Combine with RAG for knowledge-based conversations")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("   1. Authenticate: gcloud auth application-default login")
        print("   2. Install packages:")
        print("      pip install langchain langchain-google-vertexai")
        print("      pip install langchain-community")
        print("   3. Verify GCP project and Vertex AI are enabled")
        print()
        import traceback
        traceback.print_exc()
