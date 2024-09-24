

import streamlit as st
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
# from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import YoutubeLoader
from dotenv import load_dotenv

load_dotenv()

def clear_history():
    if 'history' in st.session_state:
        del st.session_state['history']

st.title("Chat with Youtube")
youtube_url = st.text_input("Input the youtube url")

if youtube_url:
    with st.spinner("Reading, Chunking and Embedding..."):
        loader = YoutubeLoader.from_youtube_url(youtube_url)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        chunks = text_splitter.split_documents(documents)

        embeddings = OpenAIEmbeddings()

        vector_store = Chroma.from_documents(chunks, embeddings)

        # llm = OpenAI(temperature=0)
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=1)

        retriever = vector_store.as_retriever()

        # chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

        crc = ConversationalRetrievalChain.from_llm(llm, retriever=retriever)
        st.session_state['crc'] = crc
        st.success("Video Loaded successfully")

question = st.text_input("Ask a question about the constitution")

if question:
    # response = chain.run(question)
    if 'crc' in st.session_state:
        crc = st.session_state['crc']
        if 'history' not in st.session_state:
            st.session_state['history'] = []
        
        response = crc.run({'question': question, 'chat_history': st.session_state['history']})
        
        st.session_state['history'].append((question, response))
        st.write(response)

        # st.write(st.session_state['history'])
        for prompts in st.session_state['history']:
            st.write("question: ", prompts[0])
            st.write("answer: ", prompts[1])
