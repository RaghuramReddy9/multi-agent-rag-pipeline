from utils.groq_client import get_groq_client

def classify_query_with_groq(query: str) -> str:
    """
    Use Groq LLM to classify which department should handle the query.
    Returns one of: 'billing', 'technical', or 'general'
    """
    client = get_groq_client()

    system_prompt = (
        "You are a support-request classifier for a company. "
        "Read the user's message and choose exactly one label:\n"
        "1. billing – if the message mentions payment, invoice, refund, subscription, or cost.\n"
        "2. technical – if it mentions bugs, errors, login, setup, or performance issues.\n"
        "3. general – for all other human-resources, account, or general questions.\n"
        "Return only the label word."
    )

    # single Groq completion call
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",           # or 'llama3-70b' / 'gemma-2b'
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.0,
        max_tokens=5,
    )

    label = completion.choices[0].message.content.strip().lower()
    if "bill" in label:
        return "billing"
    elif "tech" in label:
        return "technical"
    else:
        return "general"