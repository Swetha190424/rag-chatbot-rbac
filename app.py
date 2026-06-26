import os
import streamlit as st

# Monitoring
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-langsmith-api-key-here"   # paste langsmith key
os.environ["LANGCHAIN_PROJECT"] = "rag-resume-project"

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from qdrant_client.models import models

# ---- CONFIG ----
GROQ_API_KEY = "paste groq key"  # paste groq key

USERS = {
    "alice": {"password": "hr123",  "role": "hr"},
    "bob":   {"password": "fin123", "role": "finance"},
    "carol": {"password": "ceo123", "role": "ceo"},
}

ROLE_ACCESS = {
    "hr":      ["hr", "general"],
    "finance": ["finance", "general"],
    "ceo":     ["hr", "finance", "general"],
}

# ---- SETUP (runs once) ----
@st.cache_resource
def setup_rag():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    all_chunks = []

    files = {
        "hr": "hr_data.txt",
        "finance": "finance_data.txt",
        "general": "general_data.txt",
    }

    for dept, filename in files.items():
        loader = TextLoader(filename)
        docs = loader.load()
        chunks = splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["department"] = dept
        all_chunks.extend(chunks)

    vector_store = QdrantVectorStore.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        location=":memory:",
        collection_name="company_rbac"
    )

    llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
    analyzer = AnalyzerEngine()
    anonymizer = AnonymizerEngine()

    return vector_store, llm, analyzer, anonymizer

def is_relevant(question, llm):
    prompt = ChatPromptTemplate.from_template("""
You are a security filter for a company chatbot.
The chatbot ONLY answers questions about HR policies, employee data, finance reports, company information.
Is this question relevant? Question: "{question}"
Reply with ONLY one word: YES or NO
""")
    chain = prompt | llm
    response = chain.invoke({"question": question})
    return "YES" in response.content.strip().upper()

def mask_pii(text, analyzer, anonymizer):
    results = analyzer.analyze(
        text=text,
        entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
        language="en"
    )
    if results:
        return anonymizer.anonymize(text=text, analyzer_results=results).text
    return text

def ask_question(question, role, vector_store, llm, analyzer, anonymizer):
    allowed = ROLE_ACCESS[role]
    results = vector_store.similarity_search(
        question, k=3,
        filter=models.Filter(must=[
            models.FieldCondition(
                key="metadata.department",
                match=models.MatchAny(any=allowed)
            )
        ])
    )
    context = "\n".join([r.page_content for r in results])
    prompt = ChatPromptTemplate.from_template("""
You are a helpful company assistant.
Answer using ONLY the context below.
If not in context, say "I don't have that information."

Context: {context}
Question: {question}
Answer:
""")
    chain = prompt | llm
    response = chain.invoke({"context": context, "question": question})
    return mask_pii(response.content, analyzer, anonymizer)

# ---- UI ----
st.title("🏢 AtliQ Company Chatbot")

# Login
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    st.subheader("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = USERS.get(username)
        if user and user["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = user["role"]
            st.session_state.messages = []
            st.rerun()
        else:
            st.error("Invalid credentials!")
else:
    role = st.session_state.role
    username = st.session_state.username

    st.success(f"Welcome {username}! Role: {role.upper()}")
    st.info(f"You have access to: {ROLE_ACCESS[role]}")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()

    # Load RAG
    vector_store, llm, analyzer, anonymizer = setup_rag()

    # Chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if question := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            # Input guardrail
            if not is_relevant(question, llm):
                answer = "⚠️ Sorry, I can only answer questions about company data."
            else:
                answer = ask_question(question, role, vector_store, llm, analyzer, anonymizer)
            st.write(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})