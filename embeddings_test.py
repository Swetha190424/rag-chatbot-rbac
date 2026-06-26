from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# Step 1: Load the document
print("Loading document...")
loader = TextLoader("company_data.txt")
documents = loader.load()
print(f"Loaded {len(documents)} document")

# Step 2: Split into chunks
print("\nSplitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,      # each chunk = 200 characters
    chunk_overlap=50     # chunks overlap by 50 characters so we don't lose context
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# Step 3: Print the chunks so you can see what's happening
print("\n--- CHUNKS ---")
for i, chunk in enumerate(chunks):
    print(f"\nChunk {i+1}:")
    print(chunk.page_content)
    print("-" * 40)

# Step 4: Convert one chunk to embeddings (numbers)
print("\nConverting first chunk to embedding (numbers)...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode(chunks[0].page_content)
print(f"Embedding size: {len(embedding)} numbers")
print(f"First 5 numbers: {embedding[:5]}")