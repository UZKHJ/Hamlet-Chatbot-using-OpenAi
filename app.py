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
from google.cloud import secretmanager

#Creating a flask app and CORS Policy
app = flask.Flask(__name__)
cors = CORS(app)


# API from GCLOUD SECRET MANAGER
def access_secret_version(secret_id, version_id="latest"):
    # Create the Secret Manager client.
    client = secretmanager.SecretManagerServiceClient()

    # Build the resource name of the secret version.
    name = f"projects/advanced-deep-learning-406018/secrets/{secret_id}/versions/{version_id}"

    # Access the secret version.
    response = client.access_secret_version(name=name)

    # Return the decoded payload.
    return response.payload.data.decode('UTF-8')

OPENAI_API_KEY=access_secret_version("OPENAI_API_KEY")
if OPENAI_API_KEY:
    print("API Key loaded successfully")
else:
    print("API Key not found or invalid.")

######################################################
# FOR LOCAL WORKFLOW 
# Loads the environment variables from .env file
# load_dotenv()

# Accesses the API key using the environment variable
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
######################################################

# Splitting the document
def split_documents(documents, chunk_size=150, chunk_overlap=20):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = splitter.split_documents(documents)
    return docs

# Defining the OpenAi model to use
model_name="gpt-3.5-turbo-1106"
llm = ChatOpenAI(model_name=model_name, openai_api_key=OPENAI_API_KEY)
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