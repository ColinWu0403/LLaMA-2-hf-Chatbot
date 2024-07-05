import json
import os
import sys
import time
import pickle
import torch
import faiss
import psutil
import logging
import warnings
import numpy as np
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader
from llama_index.llms.huggingface import HuggingFaceLLM
from utils import log_time, LLM_MODEL_NAME, EMBEDDING_MODEL_NAME

# GPU acceleration with metal on Mac
device = torch.device("metal") if torch.cuda.is_available() else torch.device("cpu")

warnings.filterwarnings(
    "ignore", 
    message="Field \"model_id\" has conflict with protected namespace \"model_\"."
)

# Modify the config for pydantic models
BaseModel.model_config = {'protected_namespaces': ()}

# Function to find all PDF files in a directory and its subdirectories
def find_all_pdfs(directory):
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(root, file))
                # print(f"Found PDF file: {os.path.join(root, file)}")
    return pdf_files

def read_documents():
    # Specify the directory containing the papers
    papers_directory = "papers/"
    
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    # Get a list of all PDF files in the directory and its subdirectories
    pdf_files = find_all_pdfs(papers_directory)

    # Initialize the documents list
    documents = []

    # Batch size for processing PDF files
    batch_size = 10

    # Process PDF files in batches
    for i in range(0, len(pdf_files), batch_size):
        # if not check_memory_usage():
        #     logging.warning("Memory usage is high, pausing processing.")
        #     break
        batch = pdf_files[i:i+batch_size]
        for pdf_file in batch:
            try:
                reader = SimpleDirectoryReader(input_dir=os.path.dirname(pdf_file), required_exts=".pdf").load_data()
                documents.extend(reader)
            except Exception as e:
                logging.warning(f"Failed to read {pdf_file}: {e}")

    print(documents[0])
    return documents

# Load the vector index
def load_index(directory_path):
    index_file_path = os.path.join(directory_path, "index.pkl")

    with open(index_file_path, "rb") as f:
        index = pickle.load(f)

    log_time(f"Index loaded from {index_file_path}")
    return index


# Load the model and tokenizer
def load_model(directory_path):
    model_save_path = os.path.join(directory_path, "llm_model")
    tokenizer_save_path = os.path.join(directory_path, "llm_tokenizer")

    model = AutoModelForCausalLM.from_pretrained(model_save_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_save_path)

    log_time("Model and tokenizer loaded." + "██" * 10)
    return model, tokenizer


# Load the embedding model
def load_embedding_model(directory_path):
    embedding_model_path = os.path.join(directory_path, "embedding_model_cpu.pkl")

    with open(embedding_model_path, "rb") as f:
        embed_model = pickle.load(f)
        log_time("Embedding model loaded." + "██" * 10)
    return embed_model


# Load LLM configuration from JSON file
def load_config(directory_path):
    config_file_path = os.path.join(directory_path, "llm_config.json")

    with open(config_file_path, "r") as f:
        llm_config = json.load(f)
        log_time("LLM configuration loaded." + "██" * 20)
    return llm_config


# Import the saved model with HuggingFace
def initialize_llm(llm_config, model, tokenizer):
    llm = HuggingFaceLLM(
        context_window=llm_config["context_window"],
        max_new_tokens=llm_config["max_new_tokens"],
        generate_kwargs=llm_config["generate_kwargs"],
        system_prompt=llm_config["system_prompt"],
        query_wrapper_prompt=PromptTemplate("<|USER|>{query_str}<|ASSISTANT|>"),
        model=model,
        tokenizer=tokenizer,
        # tokenizer_name=LLM_MODEL_NAME,
        # model_name=LLM_MODEL_NAME,
        # device_map="auto",

    )

    log_time("LLM initialized.")
    return llm


# congfigure the settings
def configure_settings(embed_model, llm):
    Settings.embed_model = embed_model
    Settings.llm = llm
    Settings.chunk_size = 1024
    log_time("Settings configured.")


def generate_response(model, tokenizer, question):
    try:
        log_time(f"Using device: {device}")

        log_time("Tokenizing input...")
        inputs = tokenizer(question, return_tensors="pt").to(device)
        log_time("Input tokenized.")

        log_time("Generating response...")
        start_time = time.time()
        # output = model.generate(**inputs, max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
        )
        end_time = time.time()
        log_time(f"Response generated in {end_time - start_time:.2f} seconds.")

        log_time("Decoding response...")
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        log_time("Response decoded.")
        return response
    except Exception as e:
        log_time(f"Failed to generate response: {e}")
        return None


def initialize_all(directory):
    # index = load_index(directory)
    model, tokenizer = load_model(directory)
    embed_model = load_embedding_model(directory)
    llm_config = load_config(directory)
    llm = initialize_llm(llm_config, model, tokenizer)
    configure_settings(embed_model, llm)
    return model, tokenizer, embed_model, llm_config, llm

def get_embeddings(chunks, model, tokenizer, batch_size=8):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log_time("Get embeddings")

    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]

        # Tokenize the batch
        inputs = tokenizer(batch_chunks, padding=True, truncation=True, return_tensors="pt", max_length=512, add_special_tokens=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        
        # Extract embeddings and convert to numpy
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        
        # Store the embeddings
        all_embeddings.append(embeddings)

        # Log memory usage (optional)
        check_memory_usage()

    all_embeddings = np.vstack(all_embeddings)  # Combine all embeddings into a single array
    log_time("Embeddings retrieved.")
    
    return all_embeddings
  
def load_embed_index():
    try:
        logging.info("Loading FAISS index...")

        index = faiss.read_index('models/index_small.faiss')

        responses = np.load('models/responses_small.npy', allow_pickle=True)
        logging.info("FAISS index loaded successfully.")
        return index, responses
    except Exception as e:
        logging.error(f"Failed to load FAISS index: {e}")
        return None, None

def check_memory_usage(threshold_percent=80):
    # Get current memory usage
    memory_percent = psutil.virtual_memory().percent
    
    # Log memory usage for monitoring
    logging.info(f"Current memory usage: {memory_percent}%")
    
    # Check if memory usage exceeds the threshold
    if memory_percent > threshold_percent:
        logging.warning(f"High memory usage ({memory_percent}%), consider pausing processing.")
        return False
    
    return True

def find_best_response(text, embeddings_model, index, responses):
    # Generate embedding for the input text
    embedding = np.array(embeddings_model.embed_query(text)).reshape(1, -1)
    # print(embedding)
    
    # Query the FAISS index
    D, I = index.search(embedding, 1)  # Retrieve top 1 most similar embedding
    best_response = responses[I[0][0]]
    return best_response


# Generate answer from context:
def generate_response_from_context(model, tokenizer, question, context):
    try:
        log_time("Tokenizing input...")

        input_text = f"Question: {question}\nContext: {context}"
        inputs = tokenizer(input_text, return_tensors="pt").to(device)
        log_time("Input tokenized.")

        log_time("Generating response...")
        start_time = time.time()

        output = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=20,
            pad_token_id=tokenizer.eos_token_id,
        )
        
        end_time = time.time()
        log_time(f"Response generated in {end_time - start_time:.2f} seconds.")

        log_time("Decoding response...")
        response = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
        log_time("Response decoded.")

        return response
    except Exception as e:
        print(e)
        log_time(f"Failed to generate response: {e}")
        return None

# Initialize the model, tokenizer, and settings when the module is imported
model_dir = "./models/"
model, tokenizer, embed_model, llm_config, llm = initialize_all(model_dir)
index, responses = load_embed_index()

def runConsole():
  while True:
      input_text = input("Enter your question or type 'exit' to quit: ")
      if input_text == 'exit':
          break
      print(input_text)
      best_context = find_best_response(input_text, embed_model, index, responses)
      print("Context: " + best_context)
      response = generate_response_from_context(model, tokenizer, input_text, best_context)
      print("Answer: " + response)

# def main():
#     directory = "./models/"
#
#     index = load_index(directory)
#     model, tokenizer = load_model(directory)
#     embed_model = load_embedding_model(directory)
#     llm_config = load_config(directory)
#
#     llm = initialize_llm(llm_config, model, tokenizer)
#     configure_settings(embed_model, llm)
#
#     # Example question + response
#     question = "What is ECG (Electrocardiography)?"
#     response = generate_response(model, tokenizer, question)
#
#     print("*" * 30)
#     print("Question:", question)
#
#     if response:
#         print("Response:", response)
#     else:
#         print("Failed to generate response.")

