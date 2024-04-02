# import libraries
import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate  # Assuming this line has a typo, as `-` is not valid in module names
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from qdrant_client import QdrantClient
from langchain_community.chat_models import ChatOllama
import chainlit as cl
from langchain.chains import RetrievalQA

# Bring in our GROQ_API_KEY
from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_api_key = os.getenv("QDRANT_API_KEY")

# Custom prompt template for QA retrieval
custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful answer:
"""

def set_custom_prompt():
    """
    Function to set custom prompt template for QA retrieval
    """
    prompt = PromptTemplate(template=custom_prompt_template,
                            input_variables=["context", "question"])
    return prompt

# Initializing chat model
chat_model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768")
# chat_model = ChatGroq(temperature=0, model_name="Llama2-70b-4096")
# chat_model = ChatOllama(model="llama2", request_timeout=30.0)

# Initializing Qdrant client
client = QdrantClient(api_key=qdrant_api_key, url=qdrant_url)

def retrieval_qa_chain(llm, prompt, vectorstore):
    """
    Function to create retrieval QA chain
    """
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

def qa_bot():
    """
    Function to set up QA bot
    """
    embeddings = FastEmbedEmbeddings()
    vectorstore = Qdrant(client=client, embeddings=embeddings, collection_name="rag")
    llm = chat_model
    qa_prompt = set_custom_prompt()
    qa = retrieval_qa_chain(llm, qa_prompt, vectorstore)
    return qa

# Event handler for chat start
@cl.on_chat_start
async def start():
    """
    Initializes the bot when a new chat starts.
    """
    chain = qa_bot()
    welcome_message = cl.Message(content="Starting the bot...")
    await welcome_message.send()
    welcome_message.content = (
        "Hi, Welcome to Chat With Documents using Llamaparse, LangChain, Qdrant and models from Groq."
    )
    await welcome_message.update()
    cl.user_session.set("chain", chain)

# Event handler for incoming messages
@cl.on_message
async def main(message):
    """
    Processes incoming chat messages.
    """
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()
    cb.answer_reached = True
    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["result"]
    source_documents = res["source_documents"]

    text_elements = []  

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()
