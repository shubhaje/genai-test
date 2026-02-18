# chunking_experiment.py
import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── LLM & Embeddings (same for all experiments) ──────────────
llm = OllamaLLM(model="llama3.2")
embeddings = OllamaEmbeddings(model="llama3.2")

# ── Load docs ─────────────────────────────────────────────────
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

# ── Prompt ────────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.
ONLY answer using the context below.
If the answer is not in the context, say: "I don't know based on available information."
Never make up information.

Context: {context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── Test Questions ────────────────────────────────────────────
test_questions = [
    "What is the refund policy?",
    "How many days of annual leave do employees get?",
    "What happens in week 1 of onboarding?",
    "Can I get a refund on a digital product?",
]

# ═══════════════════════════════════════════════════════════════
# EXPERIMENT: Test 3 different chunk sizes
# ═══════════════════════════════════════════════════════════════

chunk_sizes = [200, 500, 1000]

for chunk_size in chunk_sizes:
    print("\n" + "="*60)
    print(f"EXPERIMENT: Chunk Size = {chunk_size} tokens")
    print("="*60)
    
    # 1. Create splitter with this chunk size
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=50
    )
    
    # 2. Split documents
    raw_docs = load_docs()
    chunks = splitter.split_documents(raw_docs)
    print(f"Documents split into {len(chunks)} chunks")
    
    # 3. Create fresh vector store for this experiment
    vectorstore = Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=f"chunks_{chunk_size}",  # unique name per size
        persist_directory="./chroma_experiments"
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # 4. Build RAG chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    # 5. Ask all test questions
    for q in test_questions:
        answer = rag_chain.invoke(q)
        print(f"\nQ: {q}")
        print(f"A: {answer[:150]}...")  # truncate to first 150 chars for readability
        print("-"*50)
    
    print("\n")

print("\n" + "="*60)
print("EXPERIMENT COMPLETE")
print("="*60)
print("\nNow compare the answers:")
print("- Which chunk size gave the most complete answers?")
print("- Which size had context cut off mid-sentence?")
print("- Which size retrieved irrelevant information?")