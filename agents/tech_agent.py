from utils.groq_client import get_groq_client
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os

def build_tech_vectorstore(index_path="tech_index", data_path="data/tech_faq.txt"):
    """Create or load Chroma index for technical FAQs."""
    os.makedirs(index_path, exist_ok=True)
    loader = TextLoader(data_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=index_path)
    return db


def answer_tech_query(query: str):
    """Answer tech-related queries using Groq + Chroma context."""
    client = get_groq_client()
    db = build_tech_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Retrieve context
    results = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in results])

    system_prompt = (
        "You are a technical support assistant for a SaaS product. "
        "Use the context below to answer user questions precisely and clearly. "
        "If information isn't found, politely say you don't have that data."
    )

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",   # or 'llama3-70b-8192' if you prefer consistency
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.2,
        max_tokens=150,
    )

    return completion.choices[0].message.content.strip()
