# ragas_simple.py — Fixed for Ollama
from langchain_ollama import OllamaLLM, OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from datasets import Dataset
from golden_dataset import golden_data
import os

# Import metrics from correct location
from ragas.metrics import faithfulness, answer_relevancy

print("Setting up RAG pipeline...")

# Setup LLM and embeddings
llm = OllamaLLM(model="llama3.2")
embeddings = OllamaEmbeddings(model="llama3.2")

# CRITICAL: Tell RAGAS to use Ollama for evaluation (not OpenAI)
ragas_llm = LangchainLLMWrapper(ChatOllama(model="llama3.2"))
ragas_embeddings = LangchainEmbeddingsWrapper(embeddings)

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
    collection_name="ragas_test",
    persist_directory="./chroma_ragas"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_template("""
ONLY answer using the context below.
If not in context, say: "I don't know based on available information."

Context: {context}
Question: {question}
""")

rag_chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)), 
     "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# Run RAG
print("Running RAG on questions...")
answers = []
contexts = []

for q in golden_data['question']:
    print(f"  Processing: {q[:60]}...")
    answer = rag_chain.invoke(q)
    answers.append(answer)
    contexts.append([d.page_content for d in retriever.invoke(q)])

print("\n" + "="*60)
print("Answers collected. Preparing RAGAS evaluation...")
print("="*60)

# Prepare for RAGAS
dataset = Dataset.from_dict({
    'question': golden_data['question'],
    'answer': answers,
    'contexts': contexts,
    'ground_truth': golden_data['ground_truth']
})

print("\nRunning RAGAS evaluation with Ollama...")
print("(This will take 3-5 minutes — RAGAS needs to evaluate each answer)\n")

# Evaluate with Ollama
# Evaluate with Ollama
result = evaluate(
    dataset, 
    metrics=[faithfulness, answer_relevancy],
    llm=ragas_llm,
    embeddings=ragas_embeddings
)

print("\n" + "="*70)
print("RAW RAGAS RESULTS")
print("="*70)
print(result)
print("\n")

# Try to extract scores safely
print("="*70)
print("RAGAS SCORES")
print("="*70)

try:
    # Method 1: Direct access
    faith_score = result['faithfulness']
    relevancy_score = result['answer_relevancy']
    
    print(f"Faithfulness:      {faith_score}")
    print(f"Answer Relevancy:  {relevancy_score}")
    
except Exception as e:
    print(f"Error extracting scores: {e}")
    print("\nFull result object:")
    print(result)
    print("\nResult keys:")
    print(result.keys() if hasattr(result, 'keys') else "No keys available")

print("="*70)