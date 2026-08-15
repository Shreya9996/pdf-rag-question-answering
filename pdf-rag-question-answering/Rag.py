from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

data = PyPDFLoader("Manthan-Project-Document.pdf")
docs = data.load()

embedding_model = MistralAIEmbeddings()

spiltter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = spiltter.split_documents(docs)

vector_store = Chroma.from_documents(
    documents=chunks,
    persist_directory="Ai_Project_Chroma_Database",
    embedding=embedding_model
)

retriver = vector_store.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":3}
)

model = init_chat_model("mistral-small-2603")

prompt = ChatPromptTemplate.from_messages([
    ("system",""" you are a rag agent your work is to acording to provided  user question and content 
            you should send sent relevent information by anlysing the provided content not give any ramdom answer
            give answer base on provided content only 
        """),

    ("user",""" content : {content}
               question : {question}""")    
    
]
)

print("Rag Application is Cerated ! ")

while True:

    query = input("Ask Question : ")
    if query == "0":
        break

    answer = retriver.invoke(query)

    content = "\n\n".join([doc.page_content for doc in answer])

    final_prompt = prompt.invoke({"content":content,
                                  "question" : query

                                  })

    result = model.invoke(final_prompt)

    print(f"Agent Answer : {result.content} ")







