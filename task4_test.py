from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain

# Step 1: Load the PDF
loader = PyPDFLoader("assignment.pdf")
pages = loader.load_and_split()

# Step 2: Embed and store in FAISS
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.from_documents(pages, embeddings)

# Step 3: Create retriever and LLM
retriever = db.as_retriever()
llm = Ollama(model="tinyllama")

# Step 4: QA chain for RAG
rag_chain = load_qa_chain(llm, chain_type="stuff")

# Step 5: Ask queries and compare
questions = [
    "What is RAG architecture?",
    "What is the role of vector store in RAG?",
    "Explain document chain types used in LangChain.",
    "How is RAG different from basic LLM-based QA?",
    "How does retriever help in answering questions?"
]

for i, query in enumerate(questions):
    print(f"\n🔷 Question {i+1}: {query}")

    # RAG-based answer
    docs = retriever.get_relevant_documents(query)
    rag_answer = rag_chain.run(input_documents=docs, question=query)
    
    # Raw LLM answer (without context)
    raw_answer = llm.invoke(query)

    print("📌 RAG Answer:")
    print(rag_answer)
    print("\n📌 Retrieved Chunks:")
    for d in docs:
        print("-", d.page_content[:200], "...\n")

    print("📌 Raw LLM Answer (No context):")
    print(raw_answer)
