from langchain_community.document_loaders import UnstructuredPDFLoader

#this can be used when you have the PDF file online and load it directly. 
from langchain_community.document_loaders import OnlinePDFLoader 


######### Content Processing Block ###############################

## Loading PDF file from local file directory
## read the content and store it in data object 
local_path = "Human Resources Policies and Procedures.pdf"

if local_path:
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()
else:
    print("Upload a PDF file for processing.")

#print(data[0].page_content)

## Converting content into dense vector embeddings 
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma 

#Split and chunk the data
text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
chunks = text_splitter.split_documents(data)

print("#########################Started Creating the vector Database###################")

# Add the chunks to vector database, which takes the model for creating the embeddings.
print("#####################Before Try##########################")
try:
    vector_db = Chroma.from_documents(
                                        documents=chunks, 
                                        embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
                                        collection_name="local-rag"
                                    )
    print("#####################End of Try block##########################")
except Exception as e:
    print("#####################Exception##########################")
    print(str(e))
    print("########################################################")

print("#########################Created the vector Database###################")
###################################################

######### Retrieval + Generation of Response ##############################
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

local_llm = "llama3.1"
llm = ChatOllama(model=local_llm)

QUERY_PROMPT = PromptTemplate(
    input_variables = ["question"],
    template="""You are an AI Language model assistant. Your task is to generate five different versions of the given user question to retrieve relevant documents from a vector databaase. By generating multiple perspectives on the user question, your goal is to help the user overcome some of the limitations of the distance-based similarity search. Provide these alternative questions separated by newlines. 
    Original question: {question} """
)


retriever = MultiQueryRetriever.from_llm(vector_db.as_retriever(),llm, prompt=QUERY_PROMPT)

# RAG Prompt
template = """Answer the question based ONLY on the following context: 
{context}
Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {"context":retriever, "question":RunnablePassthrough()}
    | prompt 
    | llm 
    | StrOutputParser()
)
q = "Give me completed list of categories about which poilicies are provided here."
response = chain.invoke(q)

print(response)

###################################################