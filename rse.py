from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
import streamlit as st
from dotenv import load_dotenv
import os

load_dotenv()

file_path="./faiss_index_hf"
st.title("Research Tool")
question=st.text_input("Query")
st.sidebar.title("Sources")

urls=[]
for i in range(5):
    url=st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked=st.sidebar.button("Generate Content")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if process_url_clicked:
    loader=UnstructuredURLLoader(urls=urls)
    data=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=2000,separators=["\n\n","\n"," "])
    docs=text_splitter.split_documents(data)
    vector_store=FAISS.from_documents(docs,embeddings)
    vector_store.save_local(file_path)

if question:
  llm=HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.1-8B-Instruct", 
    task="conversational",
    temperature=0.9,
    max_new_tokens=500,
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)
  llm=ChatHuggingFace(llm=llm)
  vectorstore=FAISS.load_local(folder_path=file_path,embeddings=embeddings,allow_dangerous_deserialization=True)
  chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorstore.as_retriever())
  result=chain.invoke({"question":question})
  st.subheader("Answer")
  st.text(result['answer'])
  sources=result.get("sources","")
  if sources:
     st.subheader("Sources")
     sources=sources.split("\n")
     for source in sources:
        st.write(source)





