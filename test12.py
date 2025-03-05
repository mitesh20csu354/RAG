import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import chromadb
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import os
from typing import List, Dict, Any
from constants import CHROMA_SETTINGS
from PIL import Image
import tempfile
import PyPDF4
from datetime import datetime
import io
import fitz
import json
import re
from PyPDF4 import PdfFileReader, PdfFileWriter
from PyPDF4.generic import DecodedStreamObject, EncodedStreamObject, NameObject
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from io import BytesIO
import logging
st.set_page_config(layout="wide")
# Initialize Gemini API
os.environ['GOOGLE_API_KEY'] = 'AIzaSyDefiPr8lAFdR5zLd9RmY2xXe9lVEoeydM'  # Replace with your API key
genai.configure(api_key=os.environ['GOOGLE_API_KEY'])
 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
# Constants
PERSIST_DIRECTORY = 'vectors4'
IMAGES_DIRECTORY = 'extracted_images'
text_embeddings_model_name = 'all-MiniLM-L12-v2'
 
def log_error(message):
    logging.error(message)
    st.error(message)
 
# Custom CSS for better UI
def load_css():
    st.markdown("""
        <style>
        .main {
            padding: 0rem 1rem;
        }
        .stNavigationMenu {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 10px;
        }
        .stButton button {
            width: 100%;
            border-radius: 20px;
            padding: 10px 15px;
            transition: all 0.3s ease;
        }
        .stButton button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        .css-1d391kg {
            padding: 1rem;
            border-radius: 10px;
            background-color: #ffffff;
            box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        }
        .chat-message {
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            animation: fadeIn 0.5s ease-in;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        </style>
    """, unsafe_allow_html=True)
 
def initialize_gemini():
    return genai.GenerativeModel('gemini-2.0-flash-exp')
 
def extract_text_from_pdf(file):
    pdf_reader = PyPDF4.PdfFileReader(file)
    text = []
    for page in range(pdf_reader.getNumPages()):
        text.append(pdf_reader.getPage(page).extractText())
    return "\n".join(text)
def query_gemini(model, prompt):
    try:
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        log_error(f"Error querying Gemini API: {str(e)}")
        return None
 
 
# [Previous MultiModalChatbot class remains the same]
class MultiModalChatbot:
    def __init__(self):
        try:
            self.text_embeddings = HuggingFaceEmbeddings(model_name=text_embeddings_model_name)
           
            self.chroma_client = chromadb.PersistentClient(
                path=PERSIST_DIRECTORY,
                settings=CHROMA_SETTINGS
            )
           
            self.text_db = Chroma(
                persist_directory=PERSIST_DIRECTORY,
                embedding_function=self.text_embeddings,
                collection_name="text_documents",
                client_settings=CHROMA_SETTINGS
            )
           
            self.chat_model = ChatGoogleGenerativeAI(
                model="gemini-2.0-flash-exp",
                temperature=0.2,
                convert_system_message_to_human=True
            )
           
            self.memory = ConversationBufferMemory(
                memory_key="chat_history",
                input_key="input"
            )
           
            template = """You are a helpful AI assistant with access to documents and their embedded images.
            Use the provided context to answer questions accurately and naturally.
            When user asks for an image description or question on images, refer to their specific descriptions and content , look for Days of the week , Dates & Time.
           
            Previous conversation:
            {chat_history}
           
            Context (including image descriptions):
            {context}
           
            Human: {input}
            Assistant:"""
           
            self.prompt = PromptTemplate(
                input_variables=["chat_history", "context", "input"],
                template=template
            )
           
            self.chain = LLMChain(
                llm=self.chat_model,
                prompt=self.prompt,
                memory=self.memory,
                verbose=False
            )
           
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            raise
 
    def is_image_related_query(self, query: str) -> bool:
        image_keywords = [
            'image', 'picture', 'photo', 'diagram', 'figure', 'fig',
            'visual', 'show', 'describe', 'look', 'chart', 'graph',
            'illustration', 'screenshot', 'view', 'display'
        ]
        return any(keyword.lower() in query.lower() for keyword in image_keywords)
 
    def search_knowledge_base(self, query: str, k: int = 3) -> tuple[str, List[Dict[str, Any]]]:
        try:
            image_results = []
            search_results = []
           
            if self.is_image_related_query(query):
                image_docs = self.text_db.similarity_search(
                    query,
                    k=2,
                    filter={"type": "image"}
                )
               
                for doc in image_docs:
                    if 'image_path' in doc.metadata:
                        image_results.append({
                            'path': doc.metadata['image_path'],
                            'description': doc.metadata['description'],
                            'page_number': doc.metadata.get('page_number', 'Unknown'),
                            'relevance_score': 'Direct match'
                        })
                search_results.extend(image_docs)
           
            text_docs = self.text_db.similarity_search(
                query,
                k=k,
            )
            search_results.extend(text_docs)
           
            contexts = []
            for doc in search_results:
                contexts.append(doc.page_content)
               
                if ('type' in doc.metadata and
                    doc.metadata['type'] == 'image' and
                    not any(img['path'] == doc.metadata['image_path'] for img in image_results)):
                    image_results.append({
                        'path': doc.metadata['image_path'],
                        'description': doc.metadata['description'],
                        'page_number': doc.metadata.get('page_number', 'Unknown'),
                        'relevance_score': 'Context match'
                    })
           
            return "\n".join(contexts), image_results
           
        except Exception as e:
            st.error(f"Error searching knowledge base: {str(e)}")
            return "Error retrieving context.", []
 
    def get_response(self, user_input: str) -> tuple[str, List[Dict[str, Any]]]:
        try:
            context, image_results = self.search_knowledge_base(user_input)
           
            response = self.chain.predict(
                input=user_input,
                context=context
            )
           
            return response, image_results
           
        except Exception as e:
            st.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating a response.", []
def initialize_session_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = MultiModalChatbot()
        except Exception as e:
            st.error(f"Error initializing chatbot: {str(e)}")
            st.stop()
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Chat Assistant"
 
def render_chat_assistant():

    st.image("logo1.jpg", width=80)  # Adjust width as needed
 
    
    st.header("Volkswagen Legal Buddy")
   
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(f'<div class="chat-message">{message["content"]}</div>', unsafe_allow_html=True)
            if "images" in message and message["images"]:
                cols = st.columns(min(len(message["images"]), 2))
                for idx, img in enumerate(message["images"]):
                    with cols[idx % 2]:
                        st.image(img["path"], caption=f"Page {img['page_number']}")
                        with st.expander("Image Description"):
                            st.write(img["description"])
 
    if user_input := st.chat_input("Ask me anything about your documents..."):
        with st.chat_message("user"):
            st.markdown(f'<div class="chat-message">{user_input}</div>', unsafe_allow_html=True)
        st.session_state.chat_history.append({"role": "user", "content": user_input})
       
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, image_results = st.session_state.chatbot.get_response(user_input)
                st.markdown(f'<div class="chat-message">{response}</div>', unsafe_allow_html=True)
               
                if image_results:
                    st.write("Relevant images:")
                    cols = st.columns(min(len(image_results), 2))
                    for idx, img in enumerate(image_results):
                        with cols[idx % 2]:
                            st.image(img["path"], caption=f"Page {img['page_number']}")
                            with st.expander("Image Description"):
                                st.write(img["description"])
 
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": response,
            "images": image_results
        })
 
def main():
    load_css()
   
   
    initialize_session_state()
   
    # Sidebar navigation
    with st.sidebar:
        image="image.jpg"
        st.image(image,width=60)
       
        selected_page = st.radio(
            "Select Feature",
            ["Chat Assistant"],
            key="navigation"
        )
       
        st.divider()
       
        # Add helpful information in sidebar
        if selected_page == "Chat Assistant":
            st.info("""
            💡 **Tips for Chat Assistant:**
            - Ask questions about your documents
            - Request specific image descriptions
            - Get detailed explanations
            """)
        
   
    # Main content
    if selected_page == "Chat Assistant":
        render_chat_assistant()
  
    pass
if __name__ == "__main__":
    main()