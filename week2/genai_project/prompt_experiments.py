# prompt_experiment.py
import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── Setup (same as before) ───────────────────────────────────
llm = OllamaLLM(model="llama3.2")
embeddings = OllamaEmbeddings(model="llama3.2")

def load_docs(folder="sampledocs"):
    docs = []
    for filename in os.listdir(folder):
        if filename.endswith(".txt"):
            with open(os.path.join(folder, filename)) as f:
                content = f.read()
                docs.append(Document(
                    page_content=content,
                    metadata={"source": filename}
                ))
    return docs

# Create vector store (use optimal 500 chunk size from Day 3)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
raw_docs = load_docs()
chunks = splitter.split_documents(raw_docs)

vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    collection_name="prompt_exp",
    persist_directory="./chroma_prompt_test"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ═══════════════════════════════════════════════════════════════
# TEST QUESTIONS
# ═══════════════════════════════════════════════════════════════

# 5 questions where answer IS in docs (test faithfulness)
answerable = [
    "What is the refund policy?",
    "How many days of annual leave do employees get?",
    "What happens in week 1 of onboarding?",
    "Can I get a refund on a digital product?",
    "How long is maternity leave?",
]

# 3 questions where answer is NOT in docs (test abstention)
unanswerable = [
    "What is the CEO's name?",
    "What is the company stock price?",
    "How many employees does the company have?",
]

# ═══════════════════════════════════════════════════════════════
# 3 SYSTEM PROMPTS TO TEST
# ═══════════════════════════════════════════════════════════════

prompts_to_test = {
    "NO_GUARDRAIL": """You are a helpful assistant.

Context: {context}

Question: {question}
""",
    
    "WEAK_GUARDRAIL": """You are a helpful assistant.
Try to answer using the context provided.

Context: {context}

Question: {question}
""",
    
    "STRONG_GUARDRAIL": """You are a helpful assistant.
ONLY answer using the context below.
If the answer is not in the context, say: "I don't know based on available information."
Never make up information.

Context: {context}

Question: {question}
"""
}

# ═══════════════════════════════════════════════════════════════
# RUN EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

for prompt_name, prompt_template in prompts_to_test.items():
    print("\n" + "="*70)
    print(f"EXPERIMENT: {prompt_name}")
    print("="*70)
    
    # Build RAG chain with this prompt
    prompt = ChatPromptTemplate.from_template(prompt_template)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # Test answerable questions
    print("\n--- ANSWERABLE QUESTIONS (should answer correctly) ---")
    for q in answerable:
        answer = rag_chain.invoke(q)
        print(f"\nQ: {q}")
        print(f"A: {answer[:120]}...")
        print("-"*60)
    
    # Test unanswerable questions
    print("\n--- UNANSWERABLE QUESTIONS (should say 'I don't know') ---")
    for q in unanswerable:
        answer = rag_chain.invoke(q)
        print(f"\nQ: {q}")
        print(f"A: {answer[:120]}...")
        
        # Check if it properly abstained
        abstain_phrases = ["don't know", "not available", "no information", "not in"]
        if any(phrase in answer.lower() for phrase in abstain_phrases):
            print("✅ ABSTAINED (good)")
        else:
            print("❌ HALLUCINATED (bad)")
        print("-"*60)

print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70)
print("\nAnalyze your results:")
print("- Which prompt gave the most accurate answers for answerable questions?")
print("- Which prompt properly abstained on unanswerable questions?")
print("- Did NO_GUARDRAIL hallucinate? Did WEAK_GUARDRAIL help?")