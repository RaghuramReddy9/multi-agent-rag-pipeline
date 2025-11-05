import streamlit as st
from router import classify_query_with_groq
from agents.billing_agent import answer_billing_query
from agents.tech_agent import answer_tech_query
from agents.general_agent import answer_general_query

st.set_page_config(page_title="Customer Support Multi-Agent", page_icon="🤖", layout="wide")
st.title("🤖 Customer Support Multi-Agent Assistant")
st.markdown("Ask any question related to **billing**, **technical support**, or **general inquiries**.")

# Initialize the session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat Input
user_query = st.chat_input("Type your message here...")

if user_query:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_query})

    with st.spinner("Routing your question..."):
        department = classify_query_with_groq(user_query)

    with st.spinner(f"{department.capitalize()} agent is preparing a response..."):
        if department == "billing":
            answer = answer_billing_query(user_query)
        elif department == "technical":
            answer = answer_tech_query(user_query)
        else:
            answer = answer_general_query(user_query)

    # Add Assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "agent": department
    })

# Display conversation
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(f"**({msg['agent'].capitalize()} Agent)**:{msg['content']}")
            