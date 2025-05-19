import streamlit as st
import logging
import sys
import os
from langchain_huggingface import HuggingFaceEmbeddings # Changed
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
from langchain_community.chat_models import ChatOllama # Ollama for LLM
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_core.messages import AIMessage, HumanMessage
from langchain import hub
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
    logger.info(f"App: .env file loaded. Current working directory: {os.getcwd()}")
else:
    logger.warning(f"App: Could not load .env file from {os.getcwd()}. Using defaults or other environment variables if set.")


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT") 

_app_pinecone_index_name_env = os.getenv("PINECONE_INDEX_NAME")
if _app_pinecone_index_name_env:
    PINECONE_INDEX_NAME = _app_pinecone_index_name_env
    logger.info(f"App: Using PINECONE_INDEX_NAME from .env file: '{PINECONE_INDEX_NAME}'")
else:
    PINECONE_INDEX_NAME = "catalyst-minilm-poc" # Default index name
    logger.info(f"App: PINECONE_INDEX_NAME not found in .env file. Defaulting to: '{PINECONE_INDEX_NAME}'")

_app_embedding_model_env_val = os.getenv("EMBEDDING_MODEL")
if _app_embedding_model_env_val:
    EMBEDDING_MODEL_NAME_HF = _app_embedding_model_env_val # HuggingFace model ID
    logger.info(f"App: Using EMBEDDING_MODEL (HuggingFace ID) from .env file: '{EMBEDDING_MODEL_NAME_HF}'")
else:
    EMBEDDING_MODEL_NAME_HF = "sentence-transformers/all-MiniLM-L6-v2"
    logger.info(f"App: EMBEDDING_MODEL not found in .env file. Defaulting to HuggingFace model: '{EMBEDDING_MODEL_NAME_HF}'")


if not PINECONE_API_KEY:
    st.error("PINECONE_API_KEY not found in environment variables. Please set it in a .env file.")
    logger.error("App: PINECONE_API_KEY not found.")
    sys.exit(1)


logger.info(f"App final configuration: Embedding Model (HuggingFace): '{EMBEDDING_MODEL_NAME_HF}'.")
logger.info(f"App connecting to Pinecone index: '{PINECONE_INDEX_NAME}'.")

AVAILABLE_MODELS_OLLAMA = { # This list is for Ollama LLMs
    "Llama 3 (8B)": "gemma3:12b-it-q8_0", # O    riginal default list from your first attachment
    "Mistral (7B)": "gemma3:12b-it-q8_0",
    "Phi-3 (Medium)": "gemma3:12b-it-q8_0",
}


@st.cache_resource(show_spinner="Loading Zoho Catalyst Knowledge Base from Pinecone (HuggingFace Embeddings)...")
def load_vector_store_from_pinecone():
    logger.info(f"Attempting to connect to Pinecone index: {PINECONE_INDEX_NAME} using HuggingFace embedding model {EMBEDDING_MODEL_NAME_HF}")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"App: Using device: {device} for HuggingFace embeddings.")
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME_HF,
            model_kwargs={'device': device},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info(f"HuggingFace embeddings for app initialized with model: {EMBEDDING_MODEL_NAME_HF} on {device}")

        vectorstore = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings
        )
        logger.info(f"Successfully connected to Pinecone index '{PINECONE_INDEX_NAME}'.")
        return vectorstore
    except Exception as e:
        st.error(f"Error connecting to Pinecone index '{PINECONE_INDEX_NAME}': {e}")
        logger.error(f"Error connecting to Pinecone index: {e}", exc_info=True)
        try:
            pc_check = Pinecone(api_key=PINECONE_API_KEY) 
            indexes = [idx['name'] for idx in pc_check.list_indexes()]
            logger.info(f"Available Pinecone indexes for API key: {indexes}")
            if PINECONE_INDEX_NAME not in indexes:
                st.error(f"The index '{PINECONE_INDEX_NAME}' does not seem to exist in your Pinecone project. Please run vectorstore-builder.py first with the correct embedding model settings.")
        except Exception as pe:
            logger.error(f"Could not verify Pinecone indexes: {pe}")
        return None

@st.cache_resource(show_spinner="Initializing LLM (Ollama)...")
def get_llm(model_name_ollama):
    logger.info(f"Initializing LLM (Ollama): {model_name_ollama}")
    try:
        llm = ChatOllama(
            model=model_name_ollama,
            temperature=0.1,
            num_gpu_layers=35 
        )
        logger.info(f"ChatOllama model {model_name_ollama} initialized with num_gpu_layers=35.")
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM {model_name_ollama}: {e}")
        logger.error(f"Error initializing LLM {model_name_ollama}: {e}", exc_info=True)
        return None

@st.cache_resource(show_spinner="Setting up Catalyst Assistant Agent...")
def get_agent_executor(_llm, _retriever):
    if _llm is None or _retriever is None:
        logger.error("LLM or Retriever is None, cannot create agent.")
        return None

    logger.info("Defining tools for the ReACT agent.")
    tools = [
        Tool(
            name="ZohoCatalystDocsSearch",
            func=_retriever.invoke,
            description=(
                "Use this tool to search the Zoho Catalyst documentation. "
                "It's useful for finding information about Zoho Catalyst features, concepts, "
                "syntax, parameters, API usage, SDKs, pricing, limits, "
                "best practices, troubleshooting, or for generating code snippets and template code. "
                "Input should be a concise question or search query related to Zoho Catalyst."
            )
        )
    ]
    
    prompt_template_hub_id = "hwchase17/react-chat"
    try:
        prompt_template = hub.pull(prompt_template_hub_id)
        logger.info(f"Successfully pulled prompt '{prompt_template_hub_id}' from Langchain Hub.")
    except Exception as e:
        logger.error(f"Could not pull '{prompt_template_hub_id}' from Langchain Hub: {e}. Using a fallback basic prompt structure.")
        from langchain_core.prompts import PromptTemplate
        
        fallback_template = """Assistant is a large language model.
        {tools}
        {tool_names}
        {instructions}
        {chat_history}
        Question: {input}
        {agent_scratchpad}
        """
        prompt_template = PromptTemplate.from_template(fallback_template)

    custom_instructions = """
    You are a specialized assistant for Zoho Catalyst. Your primary goal is to answer questions and generate code related to Zoho Catalyst.
    To do this, you have access to a tool called `ZohoCatalystDocsSearch`.

    **Tool Usage Guidelines:**
    1.  **Analyze the Question:** First, carefully understand the user's question.
    2.  **Decision Point:**
        *   If the question is about Zoho Catalyst features, concepts, syntax, parameters, API usage, SDKs, pricing, limits, best practices, troubleshooting, or requests for code snippets or template code, **YOU MUST use the `ZohoCatalystDocsSearch` tool.** Formulate a clear and concise query for the tool based on the user's question.
        *   If the question is a general conversational greeting, a follow-up clarification on your previous response that doesn't require new doc search, or a topic completely unrelated to Zoho Catalyst, answer directly without using the tool.
    3.  **Tool Invocation (if needed):** If you decide to use the tool, use the exact tool name `ZohoCatalystDocsSearch`.
    4.  **Synthesize Answer:** Use the information retrieved from the `ZohoCatalystDocsSearch` tool to construct a comprehensive and accurate answer. If the tool doesn't provide sufficient information, state that you couldn't find the specific details in the documentation.
    5.  **Code Generation:** When asked to generate code, use the information from the documentation to provide accurate and relevant code snippets. Explain the code clearly.

    **Interaction Style:**
    *   Be polite and helpful.
    *   If you are unsure or the information is not in the documents, say so. Do not invent information.
    *   Think step-by-step to provide the best possible response.
    """
    final_prompt = prompt_template.partial(instructions=custom_instructions)
    
    logger.info("Creating ReACT agent.")
    agent = create_react_agent(_llm, tools, final_prompt)

    logger.info("Creating AgentExecutor.")
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
    )
    logger.info("AgentExecutor created successfully.")
    return agent_executor

st.set_page_config(page_title="Zoho Catalyst RAG POC (Pinecone/HF Embeddings)", layout="wide")
st.title("Zoho Catalyst Documentation Assistant (RAG PoC with Pinecone & HuggingFace Embeddings)")
logger.info("Streamlit app started.")

vector_store_pinecone = load_vector_store_from_pinecone()
retriever = None
if vector_store_pinecone:
    retriever = vector_store_pinecone.as_retriever(search_type="mmr", search_kwargs={"k": 5, "fetch_k": 20})
    logger.info("Retriever initialized from Pinecone with MMR search, k=5, fetch_k=20.")

st.sidebar.header("LLM Configuration (Ollama)")
selected_model_display_name = st.sidebar.selectbox(
    "Choose an LLM (from Ollama):",
    options=list(AVAILABLE_MODELS_OLLAMA.keys()), # Using the Ollama model list
    index=0 
)
selected_model_ollama_name = AVAILABLE_MODELS_OLLAMA[selected_model_display_name]
logger.info(f"User selected LLM (Ollama): {selected_model_display_name} ({selected_model_ollama_name})")

llm = get_llm(selected_model_ollama_name) # This is the Ollama LLM for generation
agent_executor = None
if llm and retriever:
    agent_executor = get_agent_executor(llm, retriever)
elif not retriever:
    st.warning("Retriever not available (Pinecone connection failed or index empty). Document search functionality will be disabled.")
    logger.warning("Retriever not available. Agent will not have document search tool.")
elif not llm:
    st.error("LLM (Ollama) could not be initialized. The assistant may not function correctly.")
    logger.error("LLM not initialized. Agent executor cannot be created.")

if "messages" not in st.session_state:
    st.session_state.messages = []
    logger.info("Chat history initialized in session state.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Ask me anything about Zoho Catalyst..."):
    logger.info(f"User input: {prompt_input}")
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    if agent_executor:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                with st.spinner("Catalyst Assistant is thinking..."):
                    chat_history_langchain = []
                    for msg in st.session_state.messages[:-1]:
                        if msg["role"] == "user":
                            chat_history_langchain.append(HumanMessage(content=msg["content"]))
                        elif msg["role"] == "assistant":
                            chat_history_langchain.append(AIMessage(content=msg["content"]))

                    logger.info(f"Invoking agent with input and chat history. History length: {len(chat_history_langchain)}")
                    response = agent_executor.invoke({
                        "input": prompt_input,
                        "chat_history": chat_history_langchain
                    })
                    full_response = response.get("output", "Sorry, I encountered an issue and couldn't provide a response.")
                    logger.info(f"Agent output: {full_response}")

            except Exception as e:
                full_response = f"An error occurred: {e}"
                logger.error(f"Error during agent invocation: {e}", exc_info=True)

            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
    else:
        st.error("The assistant is not properly configured. Please check the logs.")
        logger.error("Agent executor not available for processing user input.")

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    logger.info("Chat history cleared by user.")
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"Embedding Model: HuggingFace ({EMBEDDING_MODEL_NAME_HF.split('/')[-1]})")
st.sidebar.markdown("Vector Store: Pinecone")
st.sidebar.markdown("Built by a Master's Student for a PoC.")