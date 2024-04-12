##### LLAMAPARSE #####
from llama_parse import LlamaParse

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
#
#from groq import Groq
#from langchain_groq import ChatGroq
#
import joblib
import os
import nest_asyncio  # noqa: E402
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from qdrant_client.models import Distance, VectorParams
from langchain_experimental.text_splitter import SemanticChunker

nest_asyncio.apply()
load_dotenv()

def load_or_parse_data():
    data_file = "./data/parsed_data.pkl"
    llamaparse_api_key = os.getenv("llama_cloud_apikey")
    #if os.path.exists(data_file):
        # Load the parsed data from the file
    #    parsed_data = joblib.load(data_file)
    #else:
        # Perform the parsing step and store the result in llama_parse_documents
    parsingInstructionBenefitCoverage = """The provided document is benefit & coverage information of a health plan"""
    parser = LlamaParse(api_key=llamaparse_api_key,
                        result_type="markdown",
                        parsing_instruction=parsingInstructionBenefitCoverage,
                        max_timeout=5000,)
    llama_parse_documents = parser.load_data("./data/Capital_Selection_15.30.50_20percent_1-1-24.pdf")


    # Save the parsed data to a file
    print("Saving the parse results in .pkl format ..........")
    joblib.dump(llama_parse_documents, data_file)

    # Set the parsed data to the variable
    parsed_data = llama_parse_documents

    return parsed_data

# Create vector database
def create_vector_database():

    # Call the function to either load or parse the data
    llama_parse_documents = load_or_parse_data()
    #print(llama_parse_documents[0].text[:300])

    with open('data/output.md', 'a') as f:  # Open the file in append mode ('a')
        for doc in llama_parse_documents:
            f.write(doc.text + '\n')

    markdown_path = "./data/output.md"
    loader = UnstructuredMarkdownLoader(markdown_path)

   #loader = DirectoryLoader('data/', glob="**/*.md", show_progress=True)
    documents = loader.load()

    # Split loaded documents into chunks
    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    #docs = text_splitter.split_documents(documents)

    embedding_model_name = os.getenv("embedding_model_name")

    # Initialize Embeddings
    embed_model = FastEmbedEmbeddings(model_name=embedding_model_name)

    text_splitter = SemanticChunker(
        embed_model, breakpoint_threshold_type="percentile" # "standard_deviation", "interquartile"
    )
    docs = text_splitter.split_documents(documents)
    print("Length of documents [No. of chunks] :", len(documents))

    #len(docs)
    print(f"length of documents loaded: {len(documents)}")
    print(f"total number of document chunks generated :{len(docs)}")
    #docs[0]

    

    # Create and persist a Chroma vector database from the chunked documents
    #vs = Chroma.from_documents(
    #    documents=docs,
    #    embedding=embed_model,
    #    persist_directory="chroma_db_llamaparse1",  # Local mode with in-memory storage only
    #    collection_name="rag"
    #)

    # create the qdrant client for connecting to vector db
    qdrant_url = os.getenv("qdrant_url")
    qdrant_host = os.getenv("qdrant_host")
    qdrant_cloud_cluster_apikey = os.getenv("qdrant_cloud_cluster_apikey")
    print("qdrant_url : ", qdrant_url)
    print("qdrant_host : ", qdrant_host)
    #qdrant_client = QdrantClient(url=qdrant_url)
    qdrant_client = QdrantClient(
        qdrant_host,
        api_key=qdrant_cloud_cluster_apikey,
    )
    print("Qdrant Client created successfully")

    # create the collection
    collectionName = "benefit_coverage_vector"
    qdrant_client.recreate_collection(
        collection_name=collectionName,
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )
    print("Qdrant collection created successfully")

    vs = Qdrant.from_documents(
        docs,
        embed_model,
        url=qdrant_url,
        api_key=qdrant_cloud_cluster_apikey,
        collection_name=collectionName,
        force_recreate=True
    )
    print("Vector store created successfully")

    #query it
    #query = "what is the agend of Financial Statements for 2022 ?"
    #found_doc = qdrant.similarity_search(query, k=3)
    #print(found_doc[0][:100])
    #print(qdrant.get())

    print('Vector DB created successfully !')
    return vs,embed_model

vs,embed_model = create_vector_database()

