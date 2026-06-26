from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from qdrant_client import QdrantClient
from qdrant_client.models import models

# ---- USERS DATABASE (in real app this would be a proper database) ----
USERS = {
    "alice": {"password": "hr123",      "role": "hr"},
    "bob":   {"password": "fin123",     "role": "finance"},
    "carol": {"password": "ceo123",     "role": "ceo"},
}

# What each role can access
ROLE_ACCESS = {
    "hr":      ["hr", "general"],
    "finance": ["finance", "general"],
    "ceo":     ["hr", "finance", "general"],   # CEO sees everything
}

# ---- SETUP ----
print("Setting up RAG with RBAC...")

# Load all documents with department tags
def load_documents():
    all_chunks = []
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)

    files = {
        "hr":      "hr_data.txt",
        "finance": "finance_data.txt",
        "general": "general_data.txt",
    }

    for department, filename in files.items():
        loader = TextLoader(filename)
        docs = loader.load()
        chunks = splitter.split_documents(docs)

        # Tag each chunk with department
        for chunk in chunks:
            chunk.metadata["department"] = department

        all_chunks.extend(chunks)
        print(f"Loaded {len(chunks)} chunks from {filename}")

    return all_chunks

# Create embeddings and vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
chunks = load_documents()

vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    location=":memory:",
    collection_name="company_rbac"
)

# Setup LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key="paste your key here"   # paste your key here
)

prompt = ChatPromptTemplate.from_template("""
You are a helpful company assistant.
Answer the question using ONLY the context below.
If the answer is not in the context, say "Ihr policies
 don't have that information."

Context:
{context}

Question: {question}

Answer:
""")

# ---- LOGIN ----
def login():
    print("\n=== COMPANY CHATBOT LOGIN ===")
    username = input("Username: ")
    password = input("Password: ")

    user = USERS.get(username)
    if user and user["password"] == password:
        print(f"\nWelcome {username}! You are logged in as: {user['role'].upper()}")
        return username, user["role"]
    else:
        print("Invalid credentials!")
        return None, None

# ---- ASK QUESTION WITH RBAC ----
def ask_question(question, role):
    # Get departments this role can access
    allowed_departments = ROLE_ACCESS[role]

    # Search with department filter
    results = vector_store.similarity_search(
        question,
        k=3,
        filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="metadata.department",
                    match=models.MatchAny(any=allowed_departments)
                )
            ]
        )
    )

    if not results:
        return "No relevant information found for your access level."

    context = "\n".join([r.page_content for r in results])
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return response.content

# ---- MAIN CHAT LOOP ----
username, role = login()

if role:
    print(f"\nAccess granted to: {ROLE_ACCESS[role]}")
    print("Type 'quit' to exit\n")

    while True:
        question = input("You: ")
        if question.lower() == "quit":
            break
        answer = ask_question(question, role)
        print(f"\nAssistant: {answer}\n")