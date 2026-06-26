from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

# Step 1: Load and split document
print("Loading and splitting document...")
loader = TextLoader("company_data.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 2: Create embedding model
print("\nLoading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 3: Store chunks in Qdrant (runs locally, no account needed)
print("\nStoring chunks in Qdrant...")
vector_store = QdrantVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    location=":memory:",      # runs in memory, no setup needed
    collection_name="company_data"
)
print("Stored successfully!")

# Step 4: Test a search
print("\n--- SEARCH TEST ---")
query = "how many leave days do employees get?"
results = vector_store.similarity_search(query, k=2)

print(f"Query: {query}")
print(f"\nTop matching chunks:")
for i, result in enumerate(results):
    print(f"\nResult {i+1}:")
    print(result.page_content)