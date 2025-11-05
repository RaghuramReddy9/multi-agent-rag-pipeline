from utils.groq_client import get_groq_client
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
import os

def build_billing_vectorstore(index_path="billing_index", data_path="data/billing_faq.txt"):
    """Creates or loads Chroma index for billing documents."""
    os.makedirs(index_path, exist_ok=True)
    loader = TextLoader(data_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=index_path)
    return db

def answer_billing_query(query: str):
    """Generate billing-related answer using Groq and document context."""
    client = get_groq_client()
    db = build_billing_vectorstore()
    retriever = db.as_retriever(search_kwargs={"k":3})

    # Retrieve context from billing docs
    results = retriever.invoke(query)
    context = "\n".join([doc.page_content for doc in results])

    # Generate answer
    system_prompt = (
        "You are a helpful customer support assistant specializing in billing and payment questions. "
        "Use the following company policy information to answer accurately and clearly."
    )

    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],        
        temperature=0.2,
        max_tokens=150,
    )

    answer = completion.choices[0].message.content.strip()
    return answer

    