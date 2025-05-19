Okay, here is the complete, revised `README.md` file incorporating all the details and clarifications.

---

# Simple RAG: A Flexible Retrieval Augmented Generation Chatbot

## 1. Project Purpose

Simple RAG is a Proof-of-Concept (PoC) application demonstrating how to build a Retrieval Augmented Generation (RAG) system. This project allows you to chat with your own HTML documents by:

1.  **Ingesting Data:** Processing HTML documents from a specified local directory, including its subdirectories.
2.  **Creating a Knowledge Base:** Extracting text content from these HTML files, chunking it into manageable pieces, generating dense vector embeddings (using Hugging Face sentence-transformers), and storing these embeddings in a Pinecone vector database.
3.  **Intelligent Q&A:** Providing a Streamlit-based chat interface where users can ask questions. The system uses a Large Language Model (LLM) to understand the user's query, decide if it needs to retrieve relevant context from the knowledge base (Pinecone), and then generate an informed answer based on both its inherent knowledge and the retrieved document snippets.
4.  **Flexible LLM Backend:** The application is designed to be adaptable, allowing you to switch between different LLM providers for the generative component:
    *   Models served via Ollama.
    *   Google Gemini models via API.

The core idea is to augment the LLM's knowledge with specific information from your custom HTML document set, enabling it to answer questions accurately about content it wasn't originally trained on.

## 2. Project Setup

Follow these steps to set up and run the Simple RAG project on your local machine.

### 2.1. Prerequisites

*   Python 3.9 or higher.
*   An active internet connection (for downloading models and accessing APIs).
*   Access to a Pinecone account (a free tier is available) to create and use a vector store.
*   (Optional) An API key for Google Gemini if you intend to use it as the LLM backend.
*   (Optional) Ollama installed and running locally if you intend to use Ollama-served models as the LLM backend.
*   (Optional but Recommended for local Hugging Face models) An NVIDIA GPU with CUDA installed if you want to leverage GPU acceleration for local Hugging Face embedding generation and/or local LLM inference.

### 2.2. Create a Virtual Environment (using `venv`)

It's highly recommended to use a virtual environment to manage project dependencies and avoid conflicts with other Python projects.

1.  Navigate to your project's root directory in your terminal:
    ```bash
    cd path/to/simple-rag
    ```
2.  Create a virtual environment (commonly named `.venv`):
    ```bash
    python -m venv .venv
    ```
3.  Activate the virtual environment:
    *   **On Windows (Command Prompt or PowerShell):**
        ```bash
        .venv\Scripts\activate
        ```
    *   **On macOS/Linux (bash, zsh, etc.):**
        ```bash
        source .venv/bin/activate
        ```
    After activation, your terminal prompt should be prefixed with `(.venv)`.

### 2.3. Set Up Environment Variables

This project uses a `.env` file to manage sensitive API keys and important configuration parameters.

1.  In the root directory of the project (e.g., alongside `app.py` and `vectorstore-builder.py`), create a file named `.env`.
2.  Populate this `.env` file with the following variables, replacing the placeholder values (`"YOUR_..."`, `your-...`) with your actual credentials and desired settings:

    ```env
    PINECONE_API_KEY=YOUR_API_KEY
    PINECONE_ENVIRONMENT=YOUR_ENVIRONMENT
    PINECONE_INDEX_NAME="rag-data"
    EMBEDDING_MODEL="sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION=384
    GOOGLE_API_KEY=
    ```

    **Details:**
    *   `PINECONE_API_KEY`: Your unique API key obtained from your [Pinecone account](https://www.pinecone.io/).
    *   `PINECONE_ENVIRONMENT`: The cloud environment string associated with your Pinecone project (e.g., `gcp-starter`, `us-west1-gcp`, `aws-us-east-1`). You can find this in your Pinecone console.
    *   `PINECONE_INDEX_NAME`: A unique name for the vector index that will be created in your Pinecone project to store the document embeddings.
    *   `EMBEDDING_MODEL`: The identifier for the sentence-transformer model from Hugging Face Hub that will be used to generate embeddings for your document chunks and user queries. `sentence-transformers/all-MiniLM-L6-v2` is a good, efficient default.
    *   `EMBEDDING_DIMENSION`: The dimensionality of the vectors produced by the `EMBEDDING_MODEL`. This must match the model's specification (e.g., 384 for `all-MiniLM-L6-v2`).
    *   `GOOGLE_API_KEY`: Your API key from [Google AI Studio](https://aistudio.google.com/), required only if you intend to use the `gemini.py` application variant.
    *   `GENERATION_MODEL_HF_ID`: The Hugging Face model ID for a generative LLM, required only if you use an application variant (`app_hf_local.py`) that runs the LLM locally using the `transformers` library.

### 2.4. Install Python Packages

With your virtual environment activated, install the necessary Python packages.

1.  Ensure you have a `requirements.txt` file in your project root. A comprehensive example covering all potential LLM backends:

    **`requirements.txt`:**
    ```txt
    # Core Langchain
    langchain
    langchain-community
    langchain-core
    langchain-text-splitters
    langchainhub

    # Vector Store & Embeddings
    langchain-pinecone
    pinecone-client
    langchain-huggingface # For HuggingFaceEmbeddings

    # LLM Backends
    ollama # For Ollama integration
    langchain-google-genai # For Google Gemini integration
    transformers # For local Hugging Face LLM generation
    accelerate # Helper for transformers, especially for device mapping
    bitsandbytes # For 8-bit/4-bit quantization of local HF models (optional)

    # Application & Utilities
    streamlit
    beautifulsoup4 # For HTML parsing
    python-dotenv # For managing .env files
    tiktoken # Tokenizer used by some Langchain components

    # PyTorch (for Hugging Face models: embeddings & local LLM generation)
    # It's CRITICAL to install PyTorch correctly for your system, especially for CUDA.
    # If pip install below doesn't get GPU support, install manually:
    # pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cuXXX
    # (replace cuXXX with your CUDA version like cu118 or cu121)
    # Or, if using Conda for environment management (not venv), use Conda to install PyTorch.
    torch
    torchvision
    torchaudio
    ```

2.  Install the packages using `pip`:
    ```bash
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```
    *Note on PyTorch for GPU:* If you have an NVIDIA GPU and want to use it, ensure your drivers and CUDA toolkit are compatible with the PyTorch version being installed. If `pip install torch` installs a CPU-only version, you may need to install it manually with the correct CUDA version specifier as commented in the `requirements.txt` example.

### 2.5. Prepare Your HTML Documents for Ingestion

This RAG system is specifically designed to process **HTML files** as its primary data source.

1.  **Locate or Modify `HTML_DOCS_PATH`:**
    Open the `vectorstore-builder.py` script. Near the top, you will find a variable:
    ```python
    HTML_DOCS_PATH = "data" 
    ```
    This variable defines the root directory from which the script will load your HTML documents. The default is a folder named `data` in your project's root. You can change this path if your documents are located elsewhere.

2.  **Create the Data Directory:**
    Based on the value of `HTML_DOCS_PATH` (e.g., `"data"`), create this directory in the root of your project.

3.  **Populate with HTML Files:**
    Place all your HTML documents into the directory specified by `HTML_DOCS_PATH`.
    *   **Subdirectories are Supported:** The system will automatically find and process `.html` files within any subdirectories inside the `HTML_DOCS_PATH` directory. This allows for logical organization of your content.
    *   **Example Directory Structure (if `HTML_DOCS_PATH = "data"`):**
        ```
        simple-rag/
        ├── data/                 <-- Your HTML_DOCS_PATH
        │   ├── project_alpha_docs/
        │   │   ├── introduction.html
        │   │   └── api_reference.html
        │   ├── common_faqs.html
        │   └── release_notes/
        │       ├── v1.0.html
        │       └── v1.1.html
        ├── vectorstore-builder.py
        ├── app.py                # (or app_ollama.py, gemini.py, app_hf_local.py)
        ├── .env
        └── requirements.txt 
        ```
4.  **File Type:** The script specifically looks for files ending with the `.html` extension. Other file types within this directory will be ignored during the ingestion process.

## 3. How to Run the App

Running the application involves two main steps: first building the knowledge base (vector store), and then launching the chat interface.

### 3.1. Build the Vector Store (Knowledge Base)

This crucial step processes your HTML documents, generates their vector embeddings, and populates your Pinecone index. This creates the searchable knowledge base for the RAG system.

*   **When to Run:** You need to run this script once initially to set up the knowledge base. Subsequently, you should re-run it whenever your source HTML documents are added, modified, or removed to keep the knowledge base synchronized.

*   **Execution:**
    1.  Ensure your `venv` virtual environment is activated.
    2.  Verify that your `.env` file is correctly configured with your `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, `PINECONE_INDEX_NAME`, `EMBEDDING_MODEL`, and `EMBEDDING_DIMENSION`.
    3.  Confirm that your HTML files are located in the directory specified by the `HTML_DOCS_PATH` variable within the `vectorstore-builder.py` script.
    4.  **(Potential OpenMP Issue on Windows/Linux with Intel MKL):** If you encounter an "OMP: Error #15: Initializing libiomp5md.dll..." error when running Python scripts that use libraries like NumPy or PyTorch, you may need to set an environment variable before execution. This allows multiple OpenMP runtimes to coexist (use with understanding):
        *   **Windows CMD:** `set KMP_DUPLICATE_LIB_OK=TRUE`
        *   **Windows PowerShell:** `$env:KMP_DUPLICATE_LIB_OK="TRUE"`
        *   **macOS/Linux:** `export KMP_DUPLICATE_LIB_OK=TRUE`
    5.  Run the ingestion script from your project's root directory:
        ```bash
        python vectorstore-builder.py
        ```
    6.  Monitor the console output for progress logs (e.g., files being processed, chunks created, connection to Pinecone) and any potential errors. The first time this script runs, the Hugging Face embedding model (`sentence-transformers/all-MiniLM-L6-v2` by default) will be downloaded from the Hugging Face Hub, which may take a few minutes. The overall duration depends on the number and size of your HTML documents and your internet speed for Pinecone uploads.

### 3.2. Run the Streamlit Chat Application

Once the vector store has been successfully built and populated, you can launch the interactive chat application. This project may include different application scripts (`app_*.py`) tailored for various LLM backends.

1.  Ensure your `venv` virtual environment is activated.
2.  If you are using an LLM backend that requires a local server (like Ollama) or a specific setup (like a local Hugging Face model needing significant VRAM), ensure that service/setup is ready. For Ollama, ensure the server is running and the desired LLM (e.g., `llama3:8b`) has been pulled (`ollama pull llama3:8b`).
3.  If you needed to set `KMP_DUPLICATE_LIB_OK=TRUE` for the `vectorstore-builder.py` script, you will likely need to set it again in your terminal session before running the Streamlit app.
4.  Choose the appropriate application script based on your desired LLM backend and run it using Streamlit:

    *   **For an Ollama-based LLM (e.g., if your main `app.py` is configured for Ollama):**
        ```bash
        streamlit run app.py 
        ```
    *   **For a Google Gemini-based LLM (e.g., using a `gemini.py` script):**
        Make sure `GOOGLE_API_KEY` is set in your `.env` file.
        ```bash
        streamlit run gemini.py
        ```

5.  After executing the command, Streamlit will typically start a local web server. A new tab should automatically open in your default web browser, loading the chat interface. If it doesn't, the console output will display a local URL (usually `http://localhost:8501`) that you can manually open.
6.  You can now interact with the RAG system by typing your questions into the chat input field! The assistant will use the LLM and the knowledge retrieved from your HTML documents to provide answers.
