# tests/conftest.py
import pytest
import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
"""
pytest Configuration and Fixtures

Provides reusable fixtures for RAG pipeline testing:
- rag_pipeline: Session-scoped RAG chain (built once, reused)
- sample_questions: Golden dataset for testing
"""
@pytest.fixture(scope="session")
def rag_pipeline():
    """
    Create RAG pipeline fixture for all tests.
    
    Scope: session (created once, reused across all tests)
    
    Returns:
        RunnableSequence: Complete RAG chain
        
    Note:
        Uses test_chroma directory to avoid conflicts with main pipeline
    """
    print("\n🔧 Setting up RAG pipeline for tests...")
    
    # Setup
    llm = OllamaLLM(model="llama3.2")
    embeddings = OllamaEmbeddings(model="llama3.2")
    
    def load_docs(folder="sampledocs"):
        docs = []
        for filename in os.listdir(folder):
            if filename.endswith(".txt"):
                with open(os.path.join(folder, filename)) as f:
                    docs.append(Document(
                        page_content=f.read(),
                        metadata={"source": filename}
                    ))
        return docs
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(load_docs())
    
    vectorstore = Chroma.from_documents(
        chunks, embeddings,
        collection_name="test_collection",
        persist_directory="./test_chroma"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template("""You are a helpful assistant for answering questions about company policies.

        RULES:
        1. ONLY answer using the context below
        2. If the answer is not in the context, say: "I don't know based on available information."
        3. Never make up information
        4. Do NOT write code, functions, or scripts
        5. Do NOT answer questions unrelated to the provided policies

        Context: {context}

        Question: {question}
    """)
    
    rag_chain = (
        {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
         "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    
    print("✅ RAG pipeline ready")
    return rag_chain


@pytest.fixture
def sample_questions():
    """Golden dataset questions for testing"""
    return {
        'answerable': [
            "What is the refund policy?",
            "How many days of annual leave do employees get?",
            "What happens in week 1 of onboarding?",
        ],
        'unanswerable': [
            "What is the CEO's name?",
            "What is the company stock price?",
        ]
    }