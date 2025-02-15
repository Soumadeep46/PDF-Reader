from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from src.config import Settings

settings = Settings()

def setup_llm():
    try:
        llm = HuggingFaceHub(
            repo_id='mistralai/Mistral-7B-Instruct-v0.1',
            model_kwargs={"temperature": 0.5, "max_length": 1000},
            huggingfacehub_api_token=settings.huggingface_api_token
        )
        return llm
    except Exception as e:
        raise Exception(f"Failed to setup LLM: {str(e)}")

def chat_response(llm, user_input, context):
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="""Context: {context}
        Question: {question}
        Answer:"""
    )

    chain = LLMChain(llm=llm, prompt=prompt_template)
    response = chain.run({"context": context, "question": user_input})
    
    if "Answer:" in response:
        response = response.split("Answer:")[-1].strip()
    
    return response

def summarize_content(text):
    try:
        llm = HuggingFaceHub(
            repo_id="facebook/bart-large-cnn",
            model_kwargs={"temperature": 0.2, "max_length": 250},
            huggingfacehub_api_token=settings.huggingface_api_token
        )
        summary = llm(f"Summarize: {text}")
        return summary
    except Exception as e:
        raise Exception(f"Summarization failed: {str(e)}")

