#!/usr/bin/env python3
"""
Complete RAG (Retrieval Augmented Generation) Tutorial
=======================================================

This script demonstrates all aspects of RAG with LangChain 1.0+:
- Document Loading (CSV, PDF, HTML, TXT, URLs)
- Document Splitting
- Text Embeddings
- Vector Stores (Chroma)
- Semantic Search
- RAG Chains (RetrievalQA, ConversationalRetrievalChain)

Prerequisites:
- Python 3.10+
- gcloud auth application-default login
- GCP project with Vertex AI enabled

Usage:
    python rag_complete_tutorial.py
"""

import os
import time
import vertexai
from pathlib import Path

# LangChain imports
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    TextLoader,
    BSHTMLLoader,
    WebBaseLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import numpy as np


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ID = "terraform-prj-476214"
LOCATION = "us-central1"
MODEL_NAME = "gemini-2.0-flash-exp"
EMBEDDING_MODEL = "text-embedding-004"  # Vertex AI embedding model
KNOWLEDGE_BASE_DIR = Path("knowledge-base")
CHROMA_PERSIST_DIR = Path("docs/chroma_db")


# =============================================================================
# SETUP
# =============================================================================

def setup_vertex_ai():
    """Initialize Vertex AI."""
    print("\n" + "=" * 70)
    print("🚀 Initializing Vertex AI")
    print("=" * 70)
    
    vertexai.init(project=PROJECT_ID, location=LOCATION)
    
    # Create LLM
    llm = ChatVertexAI(
        model=MODEL_NAME,
        project=PROJECT_ID,
        location=LOCATION,
        temperature=0.3,
        max_tokens=2048,
    )
    
    # Create Embeddings
    embeddings = VertexAIEmbeddings(
        model_name=EMBEDDING_MODEL,
        project=PROJECT_ID,
        location=LOCATION
    )
    
    print(f"✅ Project: {PROJECT_ID}")
    print(f"✅ Location: {LOCATION}")
    print(f"✅ Model: {MODEL_NAME}")
    print(f"✅ Embedding Model: {EMBEDDING_MODEL}")
    print()
    
    return llm, embeddings


# =============================================================================
# PART 1: DOCUMENT LOADING
# =============================================================================

def demo_csv_loading():
    """Demonstrate loading CSV documents."""
    print("\n" + "=" * 70)
    print("📄 PART 1.1: CSV Document Loading")
    print("=" * 70)
    
    csv_file = KNOWLEDGE_BASE_DIR / "Data.csv"
    print(f"\nLoading: {csv_file}")
    
    loader = CSVLoader(file_path=str(csv_file))
    data = loader.load()
    
    print(f"✅ Loaded {len(data)} documents")
    print(f"\nFirst document preview:")
    print(data[0].page_content[:200] + "...")
    print()


def demo_html_loading():
    """Demonstrate loading HTML documents."""
    print("\n" + "=" * 70)
    print("🌐 PART 1.2: HTML Document Loading")
    print("=" * 70)
    
    html_file = KNOWLEDGE_BASE_DIR / "some_website.html"
    print(f"\nLoading: {html_file}")
    
    loader = BSHTMLLoader(str(html_file))
    data = loader.load()
    
    print(f"✅ Loaded {len(data)} documents")
    print(f"\nContent preview:")
    print(data[0].page_content[:200] + "...")
    print()


def demo_url_loading():
    """Demonstrate loading documents from URLs."""
    print("\n" + "=" * 70)
    print("🔗 PART 1.3: URL Document Loading")
    print("=" * 70)
    
    url = "https://github.com/basecamp/handbook"
    print(f"\nLoading from: {url}")
    
    try:
        loader = WebBaseLoader(url)
        docs = loader.load()
        
        print(f"✅ Loaded {len(docs)} documents")
        print(f"\nContent preview:")
        print(docs[0].page_content[:500] + "...")
    except Exception as e:
        print(f"⚠️  URL loading skipped (requires internet): {str(e)[:100]}")
    print()


def demo_pdf_loading():
    """Demonstrate loading PDF documents."""
    print("\n" + "=" * 70)
    print("📕 PART 1.4: PDF Document Loading")
    print("=" * 70)
    
    pdf_file = KNOWLEDGE_BASE_DIR / "MachineLearning-Lecture01.pdf"
    print(f"\nLoading: {pdf_file}")
    
    loader = PyPDFLoader(str(pdf_file))
    pages = loader.load()
    
    print(f"✅ Loaded {len(pages)} pages")
    print(f"\nFirst page preview:")
    print(pages[0].page_content[:300] + "...")
    print()


# =============================================================================
# PART 2: DOCUMENT SPLITTING
# =============================================================================

def demo_text_splitting():
    """Demonstrate text splitting strategies."""
    print("\n" + "=" * 70)
    print("✂️  PART 2: Document Splitting")
    print("=" * 70)
    
    # Example text
    some_text = """When writing documents, writers will use document structure to group content. \
This can convey to the reader, which idea's are related. For example, closely related ideas \
are in sentences. Similar ideas are in paragraphs. Paragraphs form a document. \n\n  \
Paragraphs are often delimited with a carriage return or two carriage returns. \
Carriage returns are the "backslash n" you see embedded in this string. \
Sentences have a period at the end, but also, have a space.\
and words are separated by space."""
    
    # Configure splitter
    chunk_size = 100
    chunk_overlap = 20
    
    print(f"\nConfiguration:")
    print(f"  Chunk size: {chunk_size}")
    print(f"  Chunk overlap: {chunk_overlap}")
    
    r_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = r_splitter.split_text(some_text)
    
    print(f"\n✅ Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        print(f"\nChunk {i}:")
        print(f"  {chunk[:80]}...")
    print()


def demo_pdf_splitting():
    """Demonstrate splitting PDF documents."""
    print("\n" + "=" * 70)
    print("📚 PART 2.2: PDF Document Splitting")
    print("=" * 70)
    
    # Load multiple PDFs
    loaders = [
        PyPDFLoader(str(KNOWLEDGE_BASE_DIR / "MachineLearning-Lecture01.pdf")),
        PyPDFLoader(str(KNOWLEDGE_BASE_DIR / "MachineLearning-Lecture02.pdf")),
        PyPDFLoader(str(KNOWLEDGE_BASE_DIR / "MachineLearning-Lecture03.pdf")),
        PyPDFLoader(str(KNOWLEDGE_BASE_DIR / "MachineLearning-Lecture04.pdf")),
    ]
    
    print(f"\nLoading {len(loaders)} PDF files...")
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    
    print(f"✅ Loaded {len(docs)} pages total")
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    
    splits = text_splitter.split_documents(docs)
    
    print(f"✅ Split into {len(splits)} chunks")
    print(f"\nFirst chunk preview:")
    print(splits[0].page_content[:200] + "...")
    print()
    
    return splits


# =============================================================================
# PART 3: TEXT EMBEDDINGS
# =============================================================================

def demo_text_embeddings(embeddings):
    """Demonstrate text embeddings and similarity."""
    print("\n" + "=" * 70)
    print("🔢 PART 3: Text Embeddings")
    print("=" * 70)
    
    # Example sentences
    sentence1 = "i like dogs"
    sentence2 = "i like canines"
    sentence3 = "the weather is ugly outside"
    sentence4 = "Dog is a good companion"
    
    print("\nTest sentences:")
    print(f"  1: '{sentence1}'")
    print(f"  2: '{sentence2}'")
    print(f"  3: '{sentence3}'")
    print(f"  4: '{sentence4}'")
    
    # Generate embeddings
    print("\nGenerating embeddings...")
    embedding1 = embeddings.embed_query(sentence1)
    embedding2 = embeddings.embed_query(sentence2)
    embedding3 = embeddings.embed_query(sentence3)
    embedding4 = embeddings.embed_query(sentence4)
    
    print(f"✅ Embedding dimension: {len(embedding1)}")
    
    # Calculate similarities (dot product)
    print("\nSimilarity scores (higher = more similar):")
    print(f"  Sentence 1 & 2 (dogs/canines):  {np.dot(embedding1, embedding2):.4f}")
    print(f"  Sentence 1 & 3 (dogs/weather):  {np.dot(embedding1, embedding3):.4f}")
    print(f"  Sentence 1 & 4 (dogs/companion): {np.dot(embedding1, embedding4):.4f}")
    print(f"  Sentence 2 & 3 (canines/weather): {np.dot(embedding2, embedding3):.4f}")
    print(f"  Sentence 2 & 4 (canines/companion): {np.dot(embedding2, embedding4):.4f}")
    print(f"  Sentence 3 & 4 (weather/companion): {np.dot(embedding3, embedding4):.4f}")
    print()


# =============================================================================
# PART 4: VECTOR STORES (CHROMA)
# =============================================================================

def demo_chroma_vectorstore_csv(embeddings):
    """Demonstrate Chroma vectorstore with CSV data."""
    print("\n" + "=" * 70)
    print("💾 PART 4.1: Chroma Vectorstore - CSV Data")
    print("=" * 70)
    
    persist_directory = str(CHROMA_PERSIST_DIR / "chroma-csv")
    
    # Remove old database
    import shutil
    if Path(persist_directory).exists():
        shutil.rmtree(persist_directory)
    
    # Load CSV
    csv_file = KNOWLEDGE_BASE_DIR / "OutdoorClothingCatalog_1000.csv"
    print(f"\nLoading: {csv_file}")
    loader = CSVLoader(file_path=str(csv_file))
    documents = loader.load()
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Create vectorstore with small batches (Vertex AI token limit: ~20K tokens)
    # Each CSV row averages ~200 tokens, so use batches of 50-100
    print(f"\nCreating Chroma vectorstore...")
    batch_size = 50  # Small batches to stay under token limit
    if len(documents) > batch_size:
        print(f"   Processing {len(documents)} docs in batches of {batch_size}...")
        vectordb = Chroma.from_documents(
            documents=documents[:batch_size],
            embedding=embeddings,
            persist_directory=persist_directory
        )
        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"   Batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}: {len(batch)} docs...")
            vectordb.add_documents(batch)
            time.sleep(1)  # Rate limiting
    else:
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    
    print(f"✅ Vectorstore created with {vectordb._collection.count()} documents")
    
    # Test similarity search
    query = "Please suggest a shirt with sunblocking"
    print(f"\nQuery: '{query}'")
    docs = vectordb.similarity_search(query, k=3)
    
    print(f"\nTop result:")
    print(docs[0].page_content[:200] + "...")
    print()
    
    vectordb.persist()
    return vectordb


def demo_chroma_vectorstore_txt(embeddings):
    """Demonstrate Chroma vectorstore with text data."""
    print("\n" + "=" * 70)
    print("📝 PART 4.2: Chroma Vectorstore - Text Data")
    print("=" * 70)
    
    persist_directory = str(CHROMA_PERSIST_DIR / "chroma-txt")
    
    # Remove old database
    import shutil
    if Path(persist_directory).exists():
        shutil.rmtree(persist_directory)
    
    # Load text file
    txt_file = KNOWLEDGE_BASE_DIR / "faq.txt"
    print(f"\nLoading: {txt_file}")
    loader = TextLoader(file_path=str(txt_file))
    documents = loader.load()
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Create vectorstore
    print(f"\nCreating Chroma vectorstore...")
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"✅ Vectorstore created with {vectordb._collection.count()} documents")
    
    # Test similarity search
    query = "How do I determine my shoe size?"
    print(f"\nQuery: '{query}'")
    docs = vectordb.similarity_search(query, k=1)
    
    print(f"\nTop result:")
    print(docs[0].page_content[:300] + "...")
    print()
    
    vectordb.persist()


# =============================================================================
# PART 5: SEMANTIC SEARCH
# =============================================================================

def demo_semantic_search(embeddings):
    """Demonstrate semantic search capabilities."""
    print("\n" + "=" * 70)
    print("🔍 PART 5: Semantic Search")
    print("=" * 70)
    
    persist_directory = str(CHROMA_PERSIST_DIR / "chroma-semantic")
    
    # Remove old database
    import shutil
    if Path(persist_directory).exists():
        shutil.rmtree(persist_directory)
    
    # Sample texts about mushrooms (with duplicates)
    texts = [
        "The Amanita phalloides has a large and imposing epigeous (aboveground) fruiting body (basidiocarp).",
        "A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.",
        "A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.",
        "A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.",
        "A mushroom with a large fruiting body is the Amanita phalloides. Some varieties are all-white.",
        "A. phalloides, a.k.a Death Cap, is one of the most poisonous of all known mushrooms.",
    ]
    
    print(f"\nCreating vectorstore with {len(texts)} texts (with duplicates)...")
    vectordb = Chroma.from_texts(
        texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    question = "Tell me about mushroom"
    print(f"\nQuery: '{question}'")
    
    # Simple similarity search
    print(f"\n--- Simple Similarity Search (k=3) ---")
    results = vectordb.similarity_search(question, k=3)
    for i, doc in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"  {doc.page_content[:80]}...")
    
    # MMR search (Maximum Marginal Relevance - reduces redundancy)
    print(f"\n--- MMR Search (k=3) ---")
    print("(Maximum Marginal Relevance - diverse results)")
    results_mmr = vectordb.max_marginal_relevance_search(question, k=3)
    for i, doc in enumerate(results_mmr, 1):
        print(f"\nResult {i}:")
        print(f"  {doc.page_content[:80]}...")
    print()


# =============================================================================
# PART 6: RAG WITH RETRIEVALQA
# =============================================================================

def demo_retrievalqa(llm, embeddings):
    """Demonstrate RAG with modern LangChain 1.0+ LCEL chain."""
    print("\n" + "=" * 70)
    print("🤖 PART 6: RAG with Retrieval Chain (LCEL)")
    print("=" * 70)
    
    persist_directory = str(CHROMA_PERSIST_DIR / "chroma-rag")
    
    # Remove old database
    import shutil
    if Path(persist_directory).exists():
        shutil.rmtree(persist_directory)
    
    # Load CSV data
    csv_file = KNOWLEDGE_BASE_DIR / "OutdoorClothingCatalog_1000.csv"
    print(f"\nLoading: {csv_file}")
    loader = CSVLoader(file_path=str(csv_file))
    documents = loader.load()
    
    print(f"✅ Loaded {len(documents)} documents")
    
    # Create vectorstore with small batches (token limit)
    print(f"\nCreating vectorstore...")
    batch_size = 50
    if len(documents) > batch_size:
        print(f"   Processing {len(documents)} docs in batches of {batch_size}...")
        vectordb = Chroma.from_documents(
            documents=documents[:batch_size],
            embedding=embeddings,
            persist_directory=persist_directory
        )
        for i in range(batch_size, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"   Batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1}: {len(batch)} docs...")
            vectordb.add_documents(batch)
            time.sleep(1)
    else:
        vectordb = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    
    print(f"✅ Vectorstore created with {vectordb._collection.count()} documents")
    
    # Create RAG chain using modern LCEL (LangChain Expression Language)
    print(f"\nCreating RAG chain using LCEL...")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    # Define prompt template
    template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create the chain using LCEL
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Ask questions
    question = "Please suggest a shirt with sunblocking"
    print(f"\nQuestion: '{question}'")
    print("⏳ Querying LLM...")
    
    answer = rag_chain.invoke(question)
    
    print(f"\n✅ Answer:")
    print(f"  {answer}")
    print()
    
    vectordb.persist()


# =============================================================================
# PART 7: CONVERSATIONAL RAG
# =============================================================================

def demo_conversational_rag(llm, embeddings):
    """Demonstrate conversational RAG with memory."""
    print("\n" + "=" * 70)
    print("💬 PART 7: Conversational RAG (with Memory)")
    print("=" * 70)
    
    persist_directory = str(CHROMA_PERSIST_DIR / "chroma-conv-rag")
    
    # Remove old database
    import shutil
    if Path(persist_directory).exists():
        shutil.rmtree(persist_directory)
    
    # Load and split PDFs
    print(f"\nLoading Machine Learning lecture PDFs...")
    loaders = [
        PyPDFLoader(str(KNOWLEDGE_BASE_DIR / "MachineLearning-Lecture01.pdf")),
        PyPDFLoader(str(KNOWLEDGE_BASE_DIR / "MachineLearning-Lecture02.pdf")),
        PyPDFLoader(str(KNOWLEDGE_BASE_DIR / "MachineLearning-Lecture03.pdf")),
        PyPDFLoader(str(KNOWLEDGE_BASE_DIR / "MachineLearning-Lecture04.pdf")),
    ]
    
    docs = []
    for loader in loaders:
        docs.extend(loader.load())
    
    print(f"✅ Loaded {len(docs)} pages")
    
    # Split documents
    print(f"\nSplitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    splits = text_splitter.split_documents(docs)
    
    print(f"✅ Split into {len(splits)} chunks")
    
    # Create vectorstore with batching (196 chunks is too many for one call)
    print(f"\nCreating vectorstore...")
    batch_size = 50
    if len(splits) > batch_size:
        print(f"   Processing {len(splits)} chunks in batches of {batch_size}...")
        vectordb = Chroma.from_documents(
            documents=splits[:batch_size],
            embedding=embeddings,
            persist_directory=persist_directory
        )
        for i in range(batch_size, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            print(f"   Batch {i//batch_size + 1}/{(len(splits)-1)//batch_size + 1}: {len(batch)} chunks...")
            vectordb.add_documents(batch)
            time.sleep(1)
    else:
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    
    print(f"✅ Vectorstore created with {vectordb._collection.count()} documents")
    
    # Create conversation memory using modern LangChain 1.0+ approach
    print(f"\nCreating conversational chain with memory...")
    history = ChatMessageHistory()
    
    # Create retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    # Manual conversation handling (LangChain 1.0+ approach)
    def ask_question(question):
        """Ask a question with conversation history."""
        # Add user question to history
        history.add_user_message(question)
        
        # Get relevant documents using invoke (modern API)
        relevant_docs = retriever.invoke(question)
        
        # Create context from documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt with history
        messages = list(history.messages)
        messages.append(HumanMessage(content=f"Context: {context}\n\nQuestion: {question}"))
        
        # Get response
        response = llm.invoke(messages)
        
        # Add response to history
        history.add_ai_message(response.content)
        
        return response.content
    
    # Conversation turns
    print(f"\n--- Conversation with RAG ---")
    
    # Turn 1
    question1 = "Is probability a class topic?"
    print(f"\nTurn 1:")
    print(f"  Q: {question1}")
    print("  ⏳ Querying...")
    time.sleep(1)  # Rate limiting
    answer1 = ask_question(question1)
    print(f"  A: {answer1}")
    
    time.sleep(2)  # Rate limiting between turns
    
    # Turn 2 - references previous context
    question2 = "Why are those prerequisites needed?"
    print(f"\nTurn 2:")
    print(f"  Q: {question2}")
    print("  ⏳ Querying...")
    time.sleep(1)  # Rate limiting
    answer2 = ask_question(question2)
    print(f"  A: {answer2}")
    
    print(f"\n✅ Conversational RAG demonstrated!")
    print(f"   The AI remembers context from previous turns.")
    print()
    
    vectordb.persist()


# =============================================================================
# PART 8: COMPLETE RAG ACTIVITY (California Tours)
# =============================================================================

def demo_complete_rag_activity(llm, embeddings):
    """Complete RAG activity with California tour packages."""
    print("\n" + "=" * 70)
    print("🎯 PART 8: Complete RAG Activity - California Tours")
    print("=" * 70)
    
    persist_directory = str(CHROMA_PERSIST_DIR / "chroma-california")
    
    # Remove old database
    import shutil
    if Path(persist_directory).exists():
        shutil.rmtree(persist_directory)
    
    # Load CSV
    csv_file = KNOWLEDGE_BASE_DIR / "california_tour_package.csv"
    print(f"\nLoading: {csv_file}")
    loader = CSVLoader(file_path=str(csv_file))
    docs = loader.load()
    
    print(f"✅ Loaded {len(docs)} documents")
    
    # Split documents
    print(f"\nSplitting documents...")
    r_text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128,
        chunk_overlap=32,
        length_function=len,
        add_start_index=True,
    )
    
    pages = r_text_splitter.split_documents(docs)
    
    print(f"✅ Split into {len(pages)} chunks")
    
    # Create vectorstore with batching
    print(f"\nCreating vectorstore...")
    batch_size = 50
    if len(pages) > batch_size:
        print(f"   Processing {len(pages)} chunks in batches of {batch_size}...")
        vectordb = Chroma.from_documents(
            documents=pages[:batch_size],
            embedding=embeddings,
            persist_directory=persist_directory
        )
        for i in range(batch_size, len(pages), batch_size):
            batch = pages[i:i + batch_size]
            print(f"   Batch {i//batch_size + 1}/{(len(pages)-1)//batch_size + 1}: {len(batch)} chunks...")
            vectordb.add_documents(batch)
            time.sleep(1)
    else:
        vectordb = Chroma.from_documents(
            documents=pages,
            embedding=embeddings,
            persist_directory=persist_directory
        )
    
    print(f"✅ Vectorstore created with {vectordb._collection.count()} documents")
    
    # Test similarity search
    question = "which sessions are about augmented reality?"
    print(f"\nSimilarity Search Query: '{question}'")
    docs_result = vectordb.similarity_search(question, k=3)
    
    print(f"\nTop 3 results:")
    for i, doc in enumerate(docs_result, 1):
        print(f"\n{i}. {doc.page_content[:150]}...")
    
    # Create RAG chain using LCEL
    print(f"\nCreating RAG chain using LCEL...")
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    
    template = """Answer the question based only on the following context:
{context}

Question: {question}

Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Ask complex question
    question2 = "Give me details on The Death Valley Survivor's Trek?"
    print(f"\nRAG Query: '{question2}'")
    print("⏳ Querying LLM...")
    time.sleep(1)  # Rate limiting
    
    answer = rag_chain.invoke(question2)
    
    print(f"\n✅ Answer:")
    print(f"  {answer}")
    print()
    
    vectordb.persist()


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Run all RAG demonstrations."""
    print("\n" + "=" * 70)
    print("🎓 Complete RAG Tutorial - LangChain 1.0+ with Vertex AI")
    print("=" * 70)
    
    print(f"\nKnowledge Base: {KNOWLEDGE_BASE_DIR}")
    print(f"Chroma DB: {CHROMA_PERSIST_DIR}")
    
    # Create directories
    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    
    # Setup
    llm, embeddings = setup_vertex_ai()
    
    # Part 1: Document Loading
    print("\n" + "🔵" * 35)
    print("SECTION 1: DOCUMENT LOADING")
    print("🔵" * 35)
    demo_csv_loading()
    demo_html_loading()
    demo_url_loading()
    demo_pdf_loading()
    
    # Part 2: Document Splitting
    print("\n" + "🟢" * 35)
    print("SECTION 2: DOCUMENT SPLITTING")
    print("🟢" * 35)
    demo_text_splitting()
    demo_pdf_splitting()
    
    # Part 3: Text Embeddings
    print("\n" + "🟡" * 35)
    print("SECTION 3: TEXT EMBEDDINGS")
    print("🟡" * 35)
    demo_text_embeddings(embeddings)
    
    # Part 4: Vector Stores
    print("\n" + "🟣" * 35)
    print("SECTION 4: VECTOR STORES")
    print("🟣" * 35)
    demo_chroma_vectorstore_csv(embeddings)
    demo_chroma_vectorstore_txt(embeddings)
    
    # Part 5: Semantic Search
    print("\n" + "🔴" * 35)
    print("SECTION 5: SEMANTIC SEARCH")
    print("🔴" * 35)
    demo_semantic_search(embeddings)
    
    print("\n⏳ Waiting 3 seconds before RAG demos (rate limiting)...")
    time.sleep(3)
    
    # Part 6: RAG with RetrievalQA
    print("\n" + "🟠" * 35)
    print("SECTION 6: RAG WITH RETRIEVALQA")
    print("🟠" * 35)
    demo_retrievalqa(llm, embeddings)
    
    print("\n⏳ Waiting 3 seconds (rate limiting)...")
    time.sleep(3)
    
    # Part 7: Conversational RAG
    print("\n" + "⚫" * 35)
    print("SECTION 7: CONVERSATIONAL RAG")
    print("⚫" * 35)
    demo_conversational_rag(llm, embeddings)
    
    print("\n⏳ Waiting 3 seconds (rate limiting)...")
    time.sleep(3)
    
    # Part 8: Complete Activity
    print("\n" + "⚪" * 35)
    print("SECTION 8: COMPLETE RAG ACTIVITY")
    print("⚪" * 35)
    demo_complete_rag_activity(llm, embeddings)
    
    # Summary
    print("\n" + "=" * 70)
    print("🎉 RAG TUTORIAL COMPLETE!")
    print("=" * 70)
    
    print("\n✅ Topics Covered:")
    print("  1. Document Loading (CSV, PDF, HTML, TXT, URLs)")
    print("  2. Document Splitting (Recursive text splitter)")
    print("  3. Text Embeddings (Vertex AI embeddings)")
    print("  4. Vector Stores (Chroma)")
    print("  5. Semantic Search (Similarity & MMR)")
    print("  6. RAG with RetrievalQA")
    print("  7. Conversational RAG (with memory)")
    print("  8. Complete RAG Application")
    
    print("\n📁 Generated Assets:")
    print(f"  - Chroma databases in: {CHROMA_PERSIST_DIR}")
    print(f"  - Knowledge base files in: {KNOWLEDGE_BASE_DIR}")
    
    print("\n💡 Next Steps:")
    print("  - Explore the generated Chroma databases")
    print("  - Try your own documents and queries")
    print("  - Experiment with different chunk sizes")
    print("  - Build custom RAG applications")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\nMake sure you have:")
        print("  1. Authenticated: gcloud auth application-default login")
        print("  2. All required packages installed")
        print("  3. Knowledge base files in ./knowledge-base/")
        print()
