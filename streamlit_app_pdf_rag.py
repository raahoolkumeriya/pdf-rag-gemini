import streamlit as st
import google.generativeai as genai
from pypdf import PdfReader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import os
os.environ["GOOGLE_API_KEY"] = "AI----------------------------------Z8"

GEMINI_API_KEY = "AI-------------------------------Z8"
# Configure your Gemini API Ke
genai.configure(api_key=GEMINI_API_KEY)

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    # Splits text into 10,000 character chunks with a 1,000 char overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
    

def main():
    st.set_page_config("Chat with PDF")
    st.header("Chat with PDF using Gemini 🚀")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        if os.path.exists("faiss_index"):
            # Load the vector store and perform similarity search
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2-preview")
            new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
            docs = new_db.similarity_search(user_question)
            
            # Generate response using Gemini
            model = genai.GenerativeModel('gemini-3-flash-preview')
            context = "\n".join([doc.page_content for doc in docs])
            prompt = f"Context:\n{context}\n\nQuestion: {user_question}\nAnswer:"
            
            response = model.generate_content(prompt)
            st.write("Reply: ", response.text)
        else:
            st.error("Please upload and process a PDF first! No index found.")
            
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()
