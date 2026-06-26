from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Step 1: Load and split document
print("Setting up RAG system...")
loader = TextLoader("company_data.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Step 2: Create embeddings and store in Qdrant
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    location=":memory:",
    collection_name="company_data"
)

# Step 3: Setup LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key="paste your key here"   # paste your key here
)

# Step 4: Create prompt template
prompt = ChatPromptTemplate.from_template("""
You are a helpful company assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {question}

Answer:
""")

# Step 5: RAG function
def ask_question(question):
    # Find relevant chunks from Qdrant
    relevant_chunks = vector_store.similarity_search(question, k=2)
    
    # Combine chunks into one context string
    context = "\n".join([chunk.page_content for chunk in relevant_chunks])
    
    # Send context + question to LLM
    chain = prompt | llm
    response = chain.invoke({
        "context": context,
        "question": question
    })
    
    return response.content

# Step 6: Chat loop
print("RAG System Ready! Type 'quit' to exit\n")
while True:
    question = input("You: ")
    if question.lower() == "quit":
        break
    answer = ask_question(question)
    print(f"\nAssistant: {answer}\n")