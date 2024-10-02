#General libraries along wiih the 
import streamlit as st
import logging
import pdfplumber, os, shutil, tempfile
from typing import List, Tuple, Dict, Any, Optional

#RAG Related libraries
#import ollama
from langchain_community.document_loaders import UnstructuredPDFLoader

from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 

from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

st.set_page_config(page_title="Chat with PDFs locally!",layout="wide")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S" )

logger = logging.getLogger(__name__)

### function to extract the model names from given info

def create_vector_db(file_upload) -> Chroma:
    logger.info(f"Creating vector DB from the file upload : {file_upload.name}")
    temp_dir = tempfile.mkdtemp()

    path = os.path.join(temp_dir, file_upload.name)

    with open(path, "wb") as f:
        f.write(file_upload.getvalue())
        logger.info(f"File saved to a temporary path: {path}")
        loader = UnstructuredPDFLoader(file_path=path)
        data = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=7500, chunk_overlap=100
    )

    chunks = text_splitter.split_documents(data)

    logger.info("Document split into chunks")

    try:

        embeddings = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)
    except Exception as e:
        print(str(e))
    vector_db = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        collection_name="myRAG"
    )

    logger.info("Vector DB created")

    shutil.rmtree(temp_dir)
    logger.info(f"Temporary directory {temp_dir}  is removed successfully.")

    return vector_db

def process_question(question:str, vector_db:Chroma, selected_model:str) -> str:

    logger.info(f"""Processing question: {question}
                 using model: {selected_model}
                """)
    
    llm = ChatOllama(model=selected_model, temperature=0)

    QUERY_PROMPT = PromptTemplate(
    input_variables = ["question"],
    template="""You are an AI Language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector databaase. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. 
    Original question: {question} """)

    retriever = MultiQueryRetriever.from_llm(vector_db.as_retriever(), llm, prompt=QUERY_PROMPT)

    template = """Answer the question based ONLY on the following context: 
    {context}
    Question: {question} 
    If you don't know the answer, just say that you dont't know, don't try to make up an answer. 
    Only provide the answer from the {context}, nothing else. 
    Add snippets of the context you used to answer the question. 
    """

    prompt = ChatPromptTemplate(template_format=template)

    chain = ( 
        {"context":retriever, "question": RunnablePassthrough()}
        | prompt
        | llm 
        | StrOutputParser()
    )

    response = chain.invoke(question)

    logger.info("Question processed and response generated")

    return response


@st.cache_data
def extract_all_pages_as_images(file_upload) -> List[Any]:
    logger.info(f"""Extracting all pages as images from file :
    {file_upload.name}""")

    pdf_pages= []
    with pdfplumber.open(file_upload) as pdf:
        pdf_pages=[page.to_image().original for page in pdf.pages]
        
    logger.info("PDF pages converted to images")
    return pdf_pages 

def delete_vector_db(vector_db:Optional[Chroma]) -> None:
    logger.info("Deleting content of Vector DB")

    if(vector_db is not None):
        vector_db.delete_collection()
        st.session_state.pop("pdf_images",None)
        st.session_state.pop("file_upload",None)
        st.session_state.pop("vector_db",None)
        st.success("Collection and temporary files deleted successfully. ")

        logger.info("Vector DB and related session state is cleared successfully.")

    else:
        st.error("No vectord database was found to delete.")
        logger.warning("Attemped to delete delete vector db failed as nothing was found.")



def main():
    st.subheader("Ollama PDF RAG playground", divider="gray", anchor=False)

    #list api of ollama gets the list of all models available in the machine
    
    #models_info = ollama.list()
    #available_models = extract_model_names(models_info)

    available_models = ["llama3.1","spotnuru_model","nomic-embed-text"]

    #Divide the page in to 2 vertical sections, left section is little thinner than the right side one
    left,right = st.columns([1.5,2])

    if  "messages" not  in st.session_state:
        st.session_state["messages"] = []

    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = None 

    if available_models:
        selected_model = right.selectbox("Pick a model available locally on your system", available_models)
    
    file_upload = left.file_uploader("Upload a PDF file", 
                                    type="pdf", 
                                    accept_multiple_files=False)

    if file_upload: 
        st.session_state["file_upload"] = file_upload
        if (st.session_state["vector_db"] is None):
            st.session_state["vector_db"] = create_vector_db(file_upload)

        pdf_pages = extract_all_pages_as_images(file_upload)

        zoom_level = left.slider(
            "Zoom Level", min_value=100, max_value=1000, value=700, step=50
        )

        with left:
            with st.container(height=410, border=True):
                for page_image in pdf_pages:
                    st.image(page_image, width=zoom_level)

    delete_collection = left.button("Clear PDF Knowledge", type="secondary")

    if delete_collection:
        delete_vector_db(st.session_state["vector_db"])

    with right:
        message_container = st.container(height=500, border=True)
        if "messages" in st.session_state:
            for message in st.session_state["messages"]:
                avatar = "🤖" if message["role"] == "assistant" else "😎"

                with message_container.chat_message(message["role"], avatar=avatar):
                    st.markdown(message["content"])

        
        if prompt := st.chat_input("Enter a prompt here..."):
            try:
                st.session_state["messages"].append({"role":"user", "content":prompt})
                message_container.chat_message("assistant",avatar="🤖")

                with message_container.chat_message("assistant",avatar="🤖"):
                    with st.spinner(":green[processing...]"):
                        if st.session_state["vector_db"] is not None:
                            response = process_question(
                                prompt, st.session_state["vector_db"],selected_model
                            )
                            st.markdown(response)
                        else:
                            st.warning("Please upload a PDF file first. ")

                if st.session_state["vector_db"] is not None:
                    st.session_state["messages"].append(
                        {"role":"assistant","content":response}
                    )

            except Exception as e:
                st.error(e,icon="⛔️")
                logger.error(f"Error processing prompt: {e}")
                
        else:
            if st.session_state["vector_db"] is None:
                st.warning("Upload a PDF file to being chat...")


def check():
    st.write("Testing Streamlit")

    if "messages" in st.session_state:
        st.success("Messages are available")
    else:
        st.session_state["messages"] = []
        st.error("There are not messages")



if __name__ == "__main__":
    main()
    #check()
