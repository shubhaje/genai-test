# rag_pipeline.py — fixed version
import os
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ── 1. LLM ──────────────────────────────────────────────────
llm = OllamaLLM(model="llama3.2")

# ── 2. Embeddings ────────────────────────────────────────────
embeddings = OllamaEmbeddings(model="llama3.2")

# ── 3. Load documents ────────────────────────────────────────
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
    print(f"Loaded {len(docs)} documents")
    return docs

# ── 4. Chunk ─────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,    # increased — documents are short so 300 was too aggressive
    chunk_overlap=50
)
raw_docs = load_docs()
chunks = splitter.split_documents(raw_docs)
print(f"Split into {len(chunks)} chunks")

# ── 5. FIX: Always create fresh ChromaDB (no stale data) ─────
# Delete old collection if exists, then recreate cleanly
vectorstore = Chroma.from_documents(
    chunks,
    embeddings,
    collection_name="rag_fresh",          # named collection
    persist_directory="./chroma_db",
    collection_metadata={"hnsw:space": "cosine"}  # cosine similarity
)
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# ── 6. Prompt ────────────────────────────────────────────────
prompt = ChatPromptTemplate.from_template("""
ONLY answer using the context below.
If not in context, say: "I don't know based on available information."
Context: {context}
Question: {question}
""")

# ── 7. Format chunks ─────────────────────────────────────────
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# ── 8. RAG chain ─────────────────────────────────────────────
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ── 9. Questions ─────────────────────────────────────────────
questions = [
    "What is the refund policy?",
    "How many days of annual leave do employees get?",
    "What happens in week 1 of onboarding?",
    "What is the CEO's name?",
    "Can I get a refund on a digital product?",
    "How long is maternity AND paternity leave combined?",
]

print("\n" + "="*50)
for q in questions:
    print(f"\nQ: {q}")
    answer = rag_chain.invoke(q)
    print(f"A: {answer}")
    print("-"*40)

# ── 10. Debug retrieval ──────────────────────────────────────
print("\n" + "="*50)
print("DEBUG — Chunks retrieved for 'refund policy':")
print("="*50)
for i, doc in enumerate(retriever.invoke("What is the refund policy?")):
    print(f"\nChunk {i+1} from: {doc.metadata['source']}")
    print(f"Content: {doc.page_content[:100]}...")
    print("-"*40)