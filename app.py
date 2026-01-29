import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import load_qa_chain

from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# ---------------- PDF PROCESSING ----------------
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
    vectorstore.save_local("faiss_index")


# ---------------- RAG CHAIN ----------------
def get_conversational_chain():
    prompt = PromptTemplate(
        template="""
        Answer the question using only the context below.
        If the answer is not present, say "Answer not found in the provided documents."

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
        input_variables=["context", "question"]
    )

    llm = ChatGroq(
        model="mixtral-8x7b-32768",
        temperature=0.2
    )

    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)


def user_input(user_question):
    embeddings = OpenAIEmbeddings()
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    st.subheader("Answer")
    st.write(response["output_text"])


# ---------------- STREAMLIT UI ----------------
def main():
    st.set_page_config(page_title="RAG PDF Chat", layout="wide")
    st.title("📄 Chat with PDFs using RAG (Groq + FAISS)")

    with st.sidebar:
        st.header("Upload PDFs")
        pdf_docs = st.file_uploader(
            "Upload PDF files",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
                return

            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                chunks = get_text_chunks(raw_text)
                get_vector_store(chunks)
                st.success("PDFs indexed successfully!")

    question = st.text_input("Ask a question about your PDFs")

    if question:
        if not os.path.exists("faiss_index"):
            st.warning("Please upload and process PDFs first.")
        else:
            with st.spinner("Generating answer..."):
                user_input(question)


if __name__ == "__main__":
    main()
