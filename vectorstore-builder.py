import os
import logging
import sys
import time
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings # Changed
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec, PodSpec
from dotenv import load_dotenv
from langchain_core.documents import Document
import multiprocessing
import torch # For checking CUDA

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(module)s - %(funcName)s - Line %(lineno)d - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

if load_dotenv():
    logger.info(f".env file loaded. Current working directory: {os.getcwd()}")
else:
    logger.warning(f"Could not load .env file from {os.getcwd()}. Using defaults or other environment variables if set.")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "catalyst-minilm-poc") # Default index name

_embedding_model_env_val = os.getenv("EMBEDDING_MODEL")
if _embedding_model_env_val:
    EMBEDDING_MODEL_NAME = _embedding_model_env_val
    logger.info(f"Using EMBEDDING_MODEL (HuggingFace ID) from .env file: '{EMBEDDING_MODEL_NAME}'")
else:
    EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info(f"EMBEDDING_MODEL not found in .env file. Defaulting to HuggingFace model: '{EMBEDDING_MODEL_NAME}'")

_embedding_dimension_env_val = os.getenv("EMBEDDING_DIMENSION")
EMBEDDING_DIMENSION = 0

if _embedding_dimension_env_val:
    try:
        EMBEDDING_DIMENSION = int(_embedding_dimension_env_val)
        logger.info(f"Using EMBEDDING_DIMENSION from .env file: {EMBEDDING_DIMENSION}")
    except ValueError:
        logger.warning(f"Invalid EMBEDDING_DIMENSION in .env: '{_embedding_dimension_env_val}'. Must be an integer. Will attempt to infer.")
        _embedding_dimension_env_val = None 

if not _embedding_dimension_env_val: 
    if EMBEDDING_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2":
        EMBEDDING_DIMENSION = 384
    else:
        logger.error(f"EMBEDDING_DIMENSION not set in .env (or was invalid) and no default is known for model '{EMBEDDING_MODEL_NAME}'. "
                     "Please set EMBEDDING_DIMENSION in your .env file or use a model with a known default dimension.")
        sys.exit(1)
    logger.info(f"Inferred/defaulted EMBEDDING_DIMENSION to: {EMBEDDING_DIMENSION} for model '{EMBEDDING_MODEL_NAME}'.")


if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT:
    logger.error("PINECONE_API_KEY or PINECONE_ENVIRONMENT not found. Please set them in a .env file or as environment variables.")
    sys.exit(1)

if not PINECONE_INDEX_NAME:
    logger.error("PINECONE_INDEX_NAME not set. Please define it in a .env file or as an environment variable.")
    sys.exit(1)

logger.info(f"Final configuration for vectorstore-builder: Embedding Model (HuggingFace): '{EMBEDDING_MODEL_NAME}', Dimension: {EMBEDDING_DIMENSION}")
logger.info(f"Target Pinecone index: '{PINECONE_INDEX_NAME}' in environment: '{PINECONE_ENVIRONMENT}'")

HTML_DOCS_PATH = "data"

def process_html_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        soup = BeautifulSoup(html_content, 'html.parser')
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        text = soup.get_text(separator=' ', strip=True)
        sanitized_text = ' '.join(text.split())

        if sanitized_text:
            return Document(page_content=sanitized_text, metadata={"source": file_path})
        else:
            return None
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}", exc_info=False)
        return None

def load_and_process_html_files_parallel(docs_path):
    logger.info(f"Starting to load and process HTML files in parallel from: {docs_path}")
    filepaths_to_process = []
    for root, _, files in os.walk(docs_path):
        for file_name in files:
            if file_name.lower().endswith(".html"):
                filepaths_to_process.append(os.path.join(root, file_name))

    if not filepaths_to_process:
        logger.warning(f"No HTML files found in {docs_path}.")
        return []

    logger.info(f"Found {len(filepaths_to_process)} HTML files to process.")
    
    num_processes = multiprocessing.cpu_count() 
    logger.info(f"Using {num_processes} processes for HTML parsing.")
    
    documents = []
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.imap_unordered(process_html_file, filepaths_to_process)
        for doc in results:
            if doc:
                documents.append(doc)
    
    logger.info(f"Successfully loaded and processed {len(documents)} HTML documents in parallel.")
    return documents

def initialize_pinecone_index(index_name, dimension):
    logger.info(f"Initializing Pinecone client for environment: {PINECONE_ENVIRONMENT}")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    active_indexes = [index_info["name"] for index_info in pc.list_indexes()]
    logger.info(f"Available Pinecone indexes: {active_indexes}")

    if index_name not in active_indexes:
        logger.info(f"Pinecone index '{index_name}' not found. Creating new index...")
        
        spec = None
        pinecone_env_lower = PINECONE_ENVIRONMENT.lower()
        common_serverless_regions_prefixes = ["aws-", "gcp-", "azure-", "us-", "eu-", "ap-"]

        if "starter" in pinecone_env_lower or \
           any(pinecone_env_lower.startswith(p) for p in common_serverless_regions_prefixes) and \
           not any(pod_sig in pinecone_env_lower for pod_sig in ["p1", "p2", "s1"]):
            
            cloud_provider = "aws" 
            region_name = PINECONE_ENVIRONMENT
            if "gcp" in pinecone_env_lower:
                cloud_provider = "gcp"
            elif "azure" in pinecone_env_lower:
                cloud_provider = "azure"
            
            if "-" in region_name and not any(c in region_name for c in ["aws", "gcp", "azure"]):
                if PINECONE_ENVIRONMENT.startswith("gcp-") or ".gcp" in PINECONE_ENVIRONMENT:
                    cloud_provider = "gcp"
                else: 
                    cloud_provider = "aws"

            logger.info(f"Attempting to use ServerlessSpec for environment '{PINECONE_ENVIRONMENT}' (cloud: {cloud_provider}, region: {region_name}).")
            spec = ServerlessSpec(
                cloud=cloud_provider,
                region=region_name
            )
        else: 
            logger.info(f"Attempting to use PodSpec for environment '{PINECONE_ENVIRONMENT}'.")
            spec = PodSpec(
                environment=PINECONE_ENVIRONMENT
            )
        
        if spec is None:
            logger.error("Could not determine Pinecone spec. Aborting index creation.")
            sys.exit(1)

        try:
            pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=spec
            )
            logger.info(f"Pinecone index '{index_name}' creation initiated with dimension {dimension}, metric 'cosine', and spec: {spec}.")
            
            max_wait_time = 300 
            wait_interval = 15 
            elapsed_time = 0
            while not pc.describe_index(index_name).status['ready']:
                if elapsed_time >= max_wait_time:
                    logger.error(f"Index '{index_name}' did not become ready within {max_wait_time} seconds. Please check Pinecone console.")
                    sys.exit(1)
                logger.info(f"Waiting for index '{index_name}' to be ready... (elapsed: {elapsed_time}s)")
                time.sleep(wait_interval)
                elapsed_time += wait_interval
            logger.info(f"Index '{index_name}' is ready.")

        except Exception as e:
            logger.error(f"Error creating Pinecone index '{index_name}': {e}", exc_info=True)
            if index_name in [index_info["name"] for index_info in pc.list_indexes()]:
                 logger.info(f"Index '{index_name}' was found after creation attempt error. Assuming it exists and is ready.")
            else:
                raise
    else:
        logger.info(f"Pinecone index '{index_name}' already exists.")
    return

def build_vectorstore_pinecone(documents, embedding_model_name_param, pinecone_index_name_param, pinecone_dimension_param):
    if not documents:
        logger.error("No documents provided to build vector store. Aborting.")
        return

    logger.info(f"Initializing HuggingFace embedding model: {embedding_model_name_param}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device} for HuggingFace embeddings.")
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name_param,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True} # Often recommended for sentence transformers
        )
        logger.info(f"Successfully initialized HuggingFace embedding model: {embedding_model_name_param} on {device}.")
    except Exception as e:
        logger.error(f"Failed to initialize HuggingFace embedding model: {e}. Ensure the model name is correct and you have an internet connection for the first download.", exc_info=True)
        raise 

    logger.info("Initializing text splitter.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, # MiniLM has a smaller context window, so smaller chunks are better
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False,
    )

    logger.info("Splitting documents into chunks.")
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents.")

    if not chunks:
        logger.error("No chunks were created from the documents. Aborting vector store creation.")
        return

    initialize_pinecone_index(pinecone_index_name_param, pinecone_dimension_param)

    logger.info(f"Adding {len(chunks)} document chunks to Pinecone index '{pinecone_index_name_param}'. This might take a while...")
    
    try:
        vectorstore = PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=pinecone_index_name_param,
            batch_size=64 # Can experiment with batch size for HuggingFaceEmbeddings + Pinecone
        )
        logger.info(f"Successfully added {len(chunks)} chunks to Pinecone index '{pinecone_index_name_param}'.")
        return vectorstore
    except Exception as e:
        logger.error(f"Error building Pinecone vector store or adding documents: {e}", exc_info=True)

if __name__ == "__main__":
    logger.info("--- Starting Data Ingestion and Pinecone Vector Store Creation (HuggingFace Embeddings) ---")
    
    docs = load_and_process_html_files_parallel(HTML_DOCS_PATH)
    
    if docs:
        build_vectorstore_pinecone(docs, EMBEDDING_MODEL_NAME, PINECONE_INDEX_NAME, EMBEDDING_DIMENSION)
    else:
        logger.warning("No documents were loaded, skipping vector store creation.")
    
    logger.info("--- Data Ingestion and Pinecone Vector Store Creation Finished ---")