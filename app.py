import gradio as gr
from dotenv import load_dotenv
import os
from langchain_community.llms import HuggingFaceEndpoint
from qdrant_client import QdrantClient
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrieval for each vectorstore
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=['context', 'question'])
    return prompt

def process_question(message, history):
    print("Entered method - process_question(), message/question = ", message)
    load_dotenv()
    # create the qdrant client for connecting to vector db
    qdrant_url = os.getenv("qdrant_url")
    print("qdrant_url : ", qdrant_url)
    qdrant_client = QdrantClient(url=qdrant_url)
    print("Qdrant Client created successfully")

    # create the collection
    collectionName = "benefit_coverage_vector"

    embedding_model_name = os.getenv("embedding_model_name")
    embed_model = FastEmbedEmbeddings(model_name=embedding_model_name)

    vectorStore = Qdrant(
        embeddings = embed_model,
        client=qdrant_client,
        collection_name=collectionName
        #    metadata_payload_key="tags",
        #   content_payload_key="content",
    )
    
    retriever=vectorStore.as_retriever(search_kwargs={'k': 3})
    prompt = set_custom_prompt()

    ########################### RESPONSE ###########################
    PromptTemplate(input_variables=['context', 'question'], template="Use the following pieces of information to answer the user's question.\nIf you don't know the answer, just say that you don't know, don't try to make up an answer.\n\nContext: {context}\nQuestion: {question}\n\nOnly return the helpful answer below and nothing else.\nHelpful answer:\n")

    hf_api_key = os.getenv("hf_token")
    model_name = os.getenv("llm_model_name")
    print("model_name = ", model_name)
    llm = HuggingFaceEndpoint(repo_id=model_name, max_length=128, temperature=0.5, huggingfacehub_api_token=hf_api_key)

    qa = RetrievalQA.from_chain_type(llm=llm,
                               chain_type="stuff",
                               retriever=retriever,
                               return_source_documents=True,
                               chain_type_kwargs={"prompt": prompt})

    response = qa.invoke({"query": message})

    return response['result']


def query_LLM_Directly_without_RAG(message, history):
    print("Entered method - query_LLM_Directly_without_RAG(), message/question = ", message)
    load_dotenv()
    hf_api_key = os.getenv("hf_token")
    model_name = os.getenv("llm_model_name")
    print("model_name = ", model_name)
    llm = HuggingFaceEndpoint(repo_id=model_name, max_length=128, temperature=0.5, huggingfacehub_api_token=hf_api_key)
    print("LLM created")
    output = llm.predict(message)
    return output

#gr.ChatInterface(random_response).launch()

gr.ChatInterface(
    process_question,
    #query_LLM_Directly_without_RAG,
    #chatbot=gr.Chatbot(height=300),
    textbox=gr.Textbox(placeholder="Ask me a questions related to benefit and coverage", container=False, scale=7),
    title="Benefit & Coverage",
    description="Benefit & Coverage",
    #theme="soft",
    examples=["Do I need a referral to see a specialist ?", "How much do I have to pay to visit a specialist for in network provider ?", 
    "What if I need an immdiate medical attention ?"],
    #cache_examples=True,
    retry_btn=None,
    undo_btn="Delete Previous",
    clear_btn="Clear",
).launch()