import streamlit as st
from rag_chain import rag_chain

st.set_page_config(page_title="Fine-Print Explainer", page_icon="📄")
st.title("📄 Fine-Print Explainer")
st.caption(
    "Ask plain-language questions about your insurance policy terms. "
    "\n\nThis explains what the document says — it does not provide legal or financial advice."
)

if "history" not in st.session_state:
    st.session_state.history = []

question = st.chat_input("Ask a question about your documents...")

for q, a in st.session_state.history:
    with st.chat_message("user"):
        st.write(q)
    with st.chat_message("assistant"):
        st.write(a)

if question:
    with st.chat_message("user"):
        st.write(question)
    with st.spinner("Retrieving and generating answer..."):
        answer = rag_chain.invoke(question)
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.history.append((question, answer))