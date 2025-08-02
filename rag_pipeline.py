from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains.question_answering import load_qa_chain

# STEP 1: Load PDF
loader = PyPDFLoader(r"C:\Users\varsh\OneDrive\Desktop\Daksh_Gupta\assignment.pdf")
documents = loader.load()

# STEP 2: Create embeddings (from Ollama model)
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# STEP 3: Store vectors in Chroma
db = Chroma.from_documents(documents, embeddings)

# STEP 4: Use LLM (TinyLlama) from Ollama
llm = Ollama(model="tinyllama")

# STEP 5: Build QA Chain
chain = load_qa_chain(llm, chain_type="stuff")

# STEP 6: Ask your query
query = "What is the assignment about?"
docs = db.similarity_search(query)
response = chain.run(input_documents=docs, question=query)

# STEP 7: Output answer
print("Answer:", response)
