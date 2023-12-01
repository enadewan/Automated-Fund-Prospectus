import streamlit as st
import os
from langchain.document_loaders import PagedPDFSplitter
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import MarkdownHeaderTextSplitter
from docx import Document
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.chat_models import ChatOpenAI
import pandas as pd
import tempfile
import re
import base64
import pandas as pd
import io
import utils
from streamlit_feedback import streamlit_feedback
import trubrics
from trubrics import Trubrics
from trubrics.integrations.streamlit import FeedbackCollector
from langchain.callbacks.manager import collect_runs
from langchain import memory as lc_memory
from langsmith import Client
from streamlit_feedback import streamlit_feedback
from st_aggrid import AgGrid
import uuid
import glob
import numpy as np
from PIL import Image
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import spacy
import en_core_web_sm
import geonamescache
from pathlib import Path


def save_file(file):
        folder = 'raw_files'
        if not os.path.exists(folder):
            os.makedirs(folder)        
        files = glob.glob(folder + '/*')
        for f in files:
            os.remove(f)                    
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path


def vectorstore(file_name):
    if file_name.endswith(".pdf"):
        loader = PagedPDFSplitter(file_name)
    elif file_name.endswith(".docx"):
        loader = Docx2txtLoader(file_name)
    elif file_name.endswith(".html"):
        loader = BSHTMLLoader(file_name) 
    else:
        print("Chatbot can only parse docx, pdf and html files")

    source_pages = loader.load_and_split()
    
    embedding_func = OpenAIEmbeddings()
    search_index = FAISS.from_documents(source_pages, embedding_func) 
    search_index.save_local("index/faiss_docs_index")     

def make_context(docs):
  context = ""
  for doc in docs:
    doc = doc.page_content +  "\n\nSource: " + doc.metadata['source']
    context = context + doc + "\n\n"
  return context
def gen_dict_extract(var, key):
    if isinstance(var, dict):
        for k, v in var.items():
            if k == key:
                yield v
            if isinstance(v, (dict, list)):
                yield from gen_dict_extract(v, key)
    elif isinstance(var, list):
        for d in var:
            yield from gen_dict_extract(d, key)
    

def main():

    uploaded_files = st.sidebar.file_uploader(label='Upload files')  

    my_bool=True    
    if not uploaded_files:
        st.error("Please upload PDF/docx/html documents to continue!")            
        st.stop()

    if "messages" in st.session_state:
        my_bool=False
    
    if (uploaded_files) and my_bool==True:                            

        llm = ChatOpenAI(model ="gpt-3.5-turbo-1106",temperature = 0)
        file_path=save_file(uploaded_files)
        st.session_state['file_path']=file_path
        #my_vectorstore=vectorstore(file_path)

    from PIL import Image
    title_container = st.container()
    col1, col2 = st.columns([2, 15])
    image = Image.open('comp1.png')
    new_image = image.resize((50, 50))
    with title_container:
        with col1:
            st.image(image, width=64)
        with col2:
            st.markdown('<h1 style="color: blue;">Compliance Bot</h1>',
                        unsafe_allow_html=True)
    st.markdown(
    """
    <style>
    .container {
        display: flex;
    }
    .logo-text {
        font-weight:700 !important;
        font-size:50px !important;
        color: #f9a01b !important;
        padding-top: 75px !important;
    }
    .logo-img {
        float:right;
    }
    </style>
    """,
    unsafe_allow_html=True)
            
    file_path=st.session_state['file_path']

    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')       
        pdf_display = f'<a href="data:application/pdf;base64,{base64_pdf}" download=\'{file_path}\'>\
        Link to Original File\
        </a>'           
        st.sidebar.markdown(pdf_display, unsafe_allow_html=True)
    if 'clicks' not in st.session_state:
        st.session_state['clicks'] = {}

    def click():
        st.session_state.clicks = True


    #report = st.sidebar.radio("Select the format for your report:",["JSON", "CSV"],index=None)
    #text=st.sidebar.text_input("Please enter the topics you want to cover in your report!")
    option=st.sidebar.button("Country Restrictions/Distributions/Authorization", key="countries",type="primary") 
    my_slot1 = st.sidebar.empty() 
    my_slot2 = st.sidebar.empty() 
    option1=st.sidebar.button("Investment classes Restrictions", key="investment",type="secondary")
    option2=st.sidebar.button("Contractual Restrictions", key="contractual",type="secondary")
    st.markdown("""<style> section[data-testid="stSidebar"] div.stButton button{
                 background-color: red; 
                 width: 400px;
                 } </style>""", unsafe_allow_html=True)


    chat_history=[]
    embedding_func = OpenAIEmbeddings()
    new_vectorstore = FAISS.load_local("index/faiss_docs_index",embedding_func)   
    memory = ConversationBufferMemory(k=10,memory_key="chat_history")
    llm = ChatOpenAI(model ="gpt-3.5-turbo-1106",temperature = 0)
    ###### Prompt Enginnering using Prompt Template  #####
    template = """
    your job is to answer the questions asked by the users. Create a final answer with references ("Page number").
    If the answer is not in the context, then say that you dont know.
    Page number of the context is written at the end of the context.
    At the end of your answer write the source of the context in the following way: \n\nPage number: (Page number)
    Chat history is also provided to you.
    If the answer is not in the context, then say that you dont know the answer.

    Context: {context}
    ---

    Chat History: {chat_history}
    Question: {question}
    Answer: Let's think step by step and give best answer possible. Use points when needed. If the answer is not in the context, then say that you dont know the answer.
    """
    print(option)
    if "my_button" not in st.session_state:
        st.session_state['my_button']=False
   
    if option==True:
        print(option)
        st.session_state['my_button']=True
        loader = PagedPDFSplitter(file_path)
        source_pages = loader.load()    
        nlp = en_core_web_sm.load()
        nlp_sent_spacy = spacy.load('en_core_web_sm')
        gc = geonamescache.GeonamesCache()
        # gets nested dictionary for countries
        countries = gc.get_countries()
        countries = [*gen_dict_extract(countries, 'name')]
        country_tags=[]
        for docs in source_pages:
            doc = nlp_sent_spacy(docs.page_content)
            for ent in doc.ents:
                
                if ent.label_ == 'GPE':
                    if ent.text in countries:
                        country_tags.append(ent.text)
        set_res = set(country_tags)         
        list_res = (list(set_res))
        #print(list_res)
        list_res=["Australia", "Austria"]
        my_response=[]        
        output = pd.DataFrame()
        authentication=[]
        reason=[]
        for country in list_res:
            template = """
                your job is to answer the questions asked by the users. Create a final answer with references ("Page Number").
                If the answer is not in the context, then say that you dont know the answer.
                At the end of your answer write the source of the context in the following way: \n\nPage Number: (Page Number)
                Context: {context}
                 ---
                Question: {question}
                Answer: Let's think step by step and give best answer possible. Use points when needed. 
                Answer should be in the following format:
                Authorization status: Yes or No  
                Reason : Details of the answer
                """
            question= "Give the authorization status of " + country
            mychosendocs=new_vectorstore.similarity_search(query=question)
            context = make_context(mychosendocs)        
            my_prompt = PromptTemplate(template=template, input_variables=["context","question"]).partial(context=context)
            llm_chain = LLMChain(prompt=my_prompt, llm=llm, verbose=False)

            ###### Getting the response for the question  #####
            response = llm_chain.run(question)
            authentication.append(response.splitlines()[0].replace("Authorization status:",""))
            if response.splitlines()[1]=="":  
                reason.append(response.splitlines()[2].replace("Reason:",""))
            else:
                reason.append(response.splitlines()[1].replace("Reason:",""))       


            #print(response)
        #print(authentication)
        my_dict={"Country":list_res,"Authorization Status":authentication,"Reason":reason}
        df=pd.DataFrame(my_dict)
        pd.set_option('display.max_colwidth', None)
        df.to_csv("countries_restrictions.csv")
        df.to_json("countries_restrictions.json")
    
    if st.session_state["my_button"]==True:  
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        df=pd.read_csv("countries_restrictions.csv")
        csv = convert_df(df)
        #col1, col2 = my_slot1.columns([5,])
        #with col1:
        my_slot1.download_button(
            label="Download CSV",
            data=csv,
            file_name='countries_restrictions.csv',
            mime='text/csv',
        )
        #with col2:            
        my_slot2.download_button(
            label="Download JSON",
            data=Path("countries_restrictions.json").read_text(),
            file_name='countries_restrictions.json',
            mime='application/json',
        )
        print(option)
        option=False
        st.dataframe(df,use_container_width=False) 

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]
        

    if "prompt_ids" not in st.session_state:
        st.session_state["prompt_ids"] = []
    #if "session_id" not in st.session_state:
        #st.session_state["session_id"] = str(uuid.uuid4())

    model = "gpt-3.5-turbo"
    tags = ["app.py"]

    openai_api_key = st.secrets.get("OPENAI_API_KEY")

    messages = st.session_state.messages

    for n, msg in enumerate(messages):
        st.chat_message(msg["role"]).write(msg["content"])                 
            
            
    if question := st.chat_input("Ask your question"):
        messages.append({"role": "user", "content": question})        
        st.chat_message("user").write(question) 
        mychosendocs=new_vectorstore.similarity_search(query=question) 
        print(mychosendocs)
        context = make_context(mychosendocs)
            
        
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            my_prompt = PromptTemplate(template=template, input_variables=["context","chat_history","question"]).partial(context=context)
            llm_chain = LLMChain(prompt=my_prompt, llm=llm, verbose=True,memory=memory)

            ###### Getting the response for the question  #####
            response = llm_chain.run(question)

            print(response)
            message_placeholder.markdown(response)


            
        messages.append({"role": "assistant", "content": response})
          




if __name__ == "__main__":
     main()