import streamlit as st
from typing import Optional
from load_ingest import ingest_docs, load_documents

st.title("Github Chat")

# Input field for the clone URL
clone_url: Optional[str] = st.text_input("Enter the Git repository clone URL:")

# Text input field for the question
question: Optional[str] = st.text_input("Ask your question here:")

# Button to trigger the query
if st.button("Get Answer"):
    if question or clone_url:
        st.write("Loading...")
        documents = load_documents(clone_url)
        st.write("Loading documents")
        answer: str = ingest_docs(question, documents)
        st.write("Answer:", answer)
    else:
        st.write("Please enter a question.")
