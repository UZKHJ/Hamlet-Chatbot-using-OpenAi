# Help from w3schools.com, In-class Example Code, JSON help from CGPT (Save and Update Conversation)
# Book from https://www.gutenberg.org/ebooks/1787 | Hamlet by Shakespeare

#Importing Packages
import flask
from flask import request
from flask_cors import CORS, cross_origin
import json
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains.question_answering import load_qa_chain
import os
import openai
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

#Creating a flask app and CORS Policy
app = flask.Flask(__name__)
cors = CORS(app)


# Loads the environment variables from .env file
load_dotenv()

# Accesses the API key using the environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Splitting the document
def split_documents(documents, chunk_size=150, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(documents)
    return docs

# Defining the OpenAi model to use
model_name="gpt-3.5-turbo-1106"
llm = ChatOpenAI(model_name=model_name)
# result = llm.predict("Hello! What is your name?")
# print(result) 

#Loads all the files in text folder
loader = DirectoryLoader("./text")
txt = loader.load()
# print("Documents:", len(txt))
docs = split_documents(txt)
# print("Splitted documents:", len(docs))

# Trains and Runs the Model
def run_model(user_input):
    embeddings = SentenceTransformerEmbeddings(model_name='all-MiniLM-L6-v2')
    db = Chroma.from_documents(docs, embeddings)
    chain = load_qa_chain(llm, chain_type="stuff")
    matching_documents = db.similarity_search(user_input)
    result = chain.run(input_documents=matching_documents, question=user_input)
    return result

# Gets the user input from the index.html and returns the response
@app.route("/")
@cross_origin()
def get_results():
    user_input = request.args.get('query')
    response = run_model(user_input)
    return response


if __name__ == "__main__":
    app.run()