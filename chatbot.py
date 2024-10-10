# Import Langchain dependencies
from langchain_community.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import streamlit as st
import google.generativeai as genai
from langchain.llms.base import LLM
import os
import tempfile

# Configure the API
genai.configure(api_key="AIzaSyAq2MDtfYcK-e4oIBiDwLkrLRLolO9LdTc")

# Create the model
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
)

def send_message_to_model(message):
    chat_session = model.start_chat(history=[])
    response = chat_session.send_message(message)
    return response.text

# Function to load all PDFs from a specific folder
@st.cache_resource
def load_pdfs_from_folder(folder_path='RAG_Data/'):
    loaders = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.pdf'):
            loaders.append(PyPDFLoader(os.path.join(folder_path, file_name)))
    
    # Create index - aka vector database
    index = VectorstoreIndexCreator(
        embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=15)
    ).from_loaders(loaders)
    return index

# Load them on up
index = load_pdfs_from_folder()

# Create a custom LLM wrapper
class GeminiLLM(LLM):
    def _call(self, prompt, stop=None):
        return send_message_to_model(prompt)
    
    @property
    def _llm_type(self) -> str:
        return "gemini"

# Create a Q&A Chain
chain = RetrievalQA.from_chain_type(
    llm=GeminiLLM(),
    chain_type='stuff',
    retriever=index.vectorstore.as_retriever(),
    input_key='question'
)

# Set up the app title
st.title('Ask WatsonX')

# Setup a session state message variable to hold all the old messages
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display all the historical messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Build a prompt input template to display the prompts
prompt = st.chat_input('Pass Your Prompt here')

# Add a PDF upload option
# uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

# If the user uploads a PDF
# if uploaded_pdf:
#     # Save the uploaded PDF to a temporary file
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
#         temp_file.write(uploaded_pdf.read())
#         temp_file_path = temp_file.name
    
#     # Load the PDF using the temporary file path
#     pdf_loader = PyPDFLoader(temp_file_path)
    
    # Rebuild the index with the uploaded PDF
# index = VectorstoreIndexCreator(
#         embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
#         text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=15)
#     ).from_loaders([pdf_loader])

# Define a system prompt to set the context for the chatbot
system_prompt = "You are an AI chatbot trained to answer questions in the healthcare domain based on relevant context. If the question is out of the healthcare domain, respond with relevant context but indicate subtly: ***>>>NOTE: THIS QUESTION MIGHT BE OUT OF MY DOMAIN!<<<***\n"

# If the user hits enter then
if prompt:
    # Display the prompt
    st.chat_message('user').markdown(prompt)
    # Store the user prompt in state
    st.session_state.messages.append({'role': 'user', 'content': prompt})
    
    # Combine system prompt with user prompt
    combined_prompt = f"{system_prompt}\nUser: {prompt}\nAssistant:"
    
    # Get the response from the chain
    response = chain.run(combined_prompt)

    # Display the response
    st.chat_message('assistant').markdown(response)
    
    # Store the LLM Response in state
    st.session_state.messages.append({'role': 'assistant', 'content': response})
