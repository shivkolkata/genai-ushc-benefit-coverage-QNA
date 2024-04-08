import random
import gradio as gr
from dotenv import load_dotenv
import os
from langchain_community.llms import HuggingFaceEndpoint

def process_question(message, history):
    print("Entered method - process_question(), message/question = ", message)
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