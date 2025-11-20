# Gemini LLM Applications

This repository contains comprehensive tutorials and applications for working with Google's Gemini API and LangChain. It focuses on conversation memory management and Retrieval-Augmented Generation (RAG) systems using LangChain 1.0+ with Google Vertex AI.

## Course Contents

### Conversation Memory Tutorial
- **File**: `conversation_memory_complete_tutorial.py`
- **Topics Covered**:
  - Basic conversation memory with ChatMessageHistory
  - Building predict() functions (replaces deprecated ConversationChain)
  - Advanced memory types: Buffer, Window, Token Buffer, Summary Buffer
  - Multi-turn conversations with context retention

### RAG Tutorials (Optional)
- **File**: `rag_basics.py` - Document loading, embeddings, vector stores, basic RAG
- **File**: `rag_with_memory.py` - Conversational RAG with memory integration

## Prerequisites

### System Requirements
- **Python**: Version 3.10 or higher (Python 3.12 recommended)
- **Operating System**: macOS, Linux, or Windows
- **Google Cloud Platform**: Active GCP account with Vertex AI enabled

### GCP Setup

1. **Create or Select a GCP Project**
   - Navigate to [Google Cloud Console](https://console.cloud.google.com/)
   - Create a new project or select an existing one
   - Note your Project ID (e.g., "terraform-prj-476214")

2. **Enable Required APIs**
   ```bash
   gcloud services enable aiplatform.googleapis.com
   ```

3. **Authentication**
   ```bash
   # Install Google Cloud SDK if not already installed
   # Visit: https://cloud.google.com/sdk/docs/install
   
   # Authenticate with your Google Cloud account
   gcloud auth application-default login
   
   # Set your default project
   gcloud config set project YOUR_PROJECT_ID
   ```

## Environment Setup

### Step 1: Create Virtual Environment

```bash
# Clone or navigate to the repository
cd gemini-llm-applications

# Create virtual environment with Python 3.12
python3.12 -m venv venv-py312

# For systems with different Python versions:
# python3.10 -m venv venv-py312
# python3.11 -m venv venv-py312
```

### Step 2: Activate Virtual Environment

**macOS/Linux:**
```bash
source venv-py312/bin/activate
```

**Windows:**
```cmd
venv-py312\Scripts\activate
```

### Step 3: Upgrade pip

```bash
pip install --upgrade pip
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- LangChain 1.0+ packages (core, community, text-splitters)
- Google Cloud Vertex AI integration
- Chroma vector database
- Document processing libraries (PyPDF, BeautifulSoup)
- Required utilities

### Step 5: Verify Installation

```bash
# Test Python version
python --version

# Test LangChain imports
python -c "from langchain_google_vertexai import ChatVertexAI; print('Success')"

# Test Google Cloud authentication
gcloud auth application-default print-access-token
```

## Running the Tutorials

### Conversation Memory Tutorial

```bash
# Ensure virtual environment is activated
source venv-py312/bin/activate  # macOS/Linux
# or
venv-py312\Scripts\activate     # Windows

# Run the complete conversation memory tutorial
python conversation_memory_complete_tutorial.py
```

This tutorial demonstrates:

- Basic conversation memory with ChatMessageHistory
- Building predict() functions for stateful conversations
- Buffer Memory: Stores all conversation messages
- Window Memory: Keeps only the last K exchanges
- Token Buffer Memory: Manages messages within token limits
- Summary Buffer Memory: Summarizes older messages while keeping recent ones

### Configuration

Update the project configuration in the tutorial files:

```python
PROJECT_ID = "your-gcp-project-id"  # Replace with your GCP project ID
LOCATION = "us-central1"            # Or your preferred region
```

## Troubleshooting

### Authentication Issues

If you encounter authentication errors:

```bash
# Re-authenticate
gcloud auth application-default login

# Verify credentials
gcloud auth application-default print-access-token

# Check active account
gcloud auth list
```

### Import Errors

If you encounter import errors:

```bash
# Reinstall LangChain packages
pip uninstall langchain langchain-core langchain-community langchain-google-vertexai -y
pip install langchain langchain-core langchain-community langchain-google-vertexai
```

### Rate Limit Errors

The tutorials include built-in rate limiting (3-second delays between API calls). If you still encounter quota errors:

- Check your Vertex AI quotas in the GCP Console
- Request quota increases if needed
- Increase the sleep delays in the code

## Documentation

- [Creating a Google AI Studio API Key](creating-google-ai-studio-api-key.md)
- [LangChain Documentation](https://python.langchain.com/)
- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)

## Additional Resources

### LangChain 1.0+ Migration

This course uses LangChain 1.0+ with modern APIs:

- No deprecated chains (ConversationChain, RetrievalQA)
- Uses LangChain Expression Language (LCEL)
- Modern prompt templates and message histories
- Compatible with latest Google Vertex AI models

### Project Structure

```
gemini-llm-applications/
|-- conversation_memory_complete_tutorial.py  # Main memory tutorial
|-- rag_basics.py                             # RAG fundamentals (optional)
|-- rag_with_memory.py                        # Advanced RAG (optional)
|-- requirements.txt                          # Python dependencies
|-- README.md                                 # This file
|-- knowledge-base/                           # Sample data files
    |-- Data.csv
    |-- *.pdf
    |-- *.html
```

## License

See LICENSE file for details.

## Support

For issues or questions:

1. Check the troubleshooting section above
2. Review the tutorial code comments
3. Consult LangChain and Vertex AI documentation
4. Contact your instructor
