from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# 🔹 1. Load your PDF
loader = PyPDFLoader("assignment.pdf")
documents = loader.load()

# 🔹 2. Convert into embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")
db = FAISS.from_documents(documents, embeddings)
retriever = db.as_retriever()

# 🔹 3. Custom prompt with bullet points, citations and disclaimer
template = """
You are an AI assistant. Use the following context to answer the question.

- Provide the answer in bullet points
- Mention citations using [source X]
- Add this at the end: "Note: This is an AI-generated response based on the provided documents."

Context:
{context}

Question: {question}

Helpful Answer:
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# 🔹 4. Load your LLM and QA Chain
llm = Ollama(model="tinyllama")

rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# 🔹 5. Ask questions
questions = [
    "What is RAG architecture?",
    "What is the use of vectorstore?",
    "How does retriever work?",
]

for query in questions:
    print(f"\n🔷 Question: {query}")
    answer = rag_chain.run(query)
    print(f"📌 Answer:\n{answer}")
