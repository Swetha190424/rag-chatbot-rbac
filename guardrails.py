import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

# ---- SETUP ----
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key="paste your key here"  # paste your key here
)

# PII detector setup
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

# ---- INPUT GUARDRAIL ----
def is_question_relevant(question):
    """Check if question is relevant to company data"""
    prompt = ChatPromptTemplate.from_template("""
You are a security filter for a company chatbot.
The chatbot ONLY answers questions about:
- HR policies
- Employee data
- Finance reports
- Company information

Is this question relevant to the above topics?
Question: "{question}"

Reply with ONLY one word: YES or NO
""")
    chain = prompt | llm
    response = chain.invoke({"question": question})
    answer = response.content.strip().upper()
    return "YES" in answer

# ---- OUTPUT GUARDRAIL ----
def mask_pii(text):
    """Detect and mask PII in the response"""
    results = analyzer.analyze(
        text=text,
        entities=["EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD"],
        language="en"
    )
    if results:
        anonymized = anonymizer.anonymize(text=text, analyzer_results=results)
        return anonymized.text
    return text

# ---- TEST IT ----
print("=== TESTING GUARDRAILS ===\n")

# Test 1: Input guardrail - relevant questions
print("--- INPUT GUARDRAIL TESTS ---")
questions = [
    "How many leave days do employees get?",   # relevant
    "What was Q2 revenue?",                    # relevant
    "Write me a poem about cats",              # NOT relevant
    "Who is Elon Musk?",                       # NOT relevant
    "What is the marketing budget?",           # relevant
]

for q in questions:
    relevant = is_question_relevant(q)
    status = "✅ ALLOWED" if relevant else "❌ BLOCKED"
    print(f"{status} → {q}")

# Test 2: Output guardrail - PII masking
print("\n--- OUTPUT GUARDRAIL TESTS ---")
test_outputs = [
    "The employee's email is john.doe@atliq.com and phone is 555-123-4567",
    "Q2 revenue was $2.8 million",
    "Contact HR at hr@atliq.com for more details",
]

for text in test_outputs:
    masked = mask_pii(text)
    print(f"\nOriginal: {text}")
    print(f"Masked:   {masked}")