import os 
import glob 
from typing import List 
from dotenv import load_dotenv 
from tqdm import tqdm 
 
 
from langchain.document_loaders import ( 
    CSVLoader, 
    EverNoteLoader, 
    PyPDFLoader, 
    TextLoader, 
    UnstructuredEmailLoader, 
    UnstructuredEPubLoader, 
    UnstructuredHTMLLoader, 
    UnstructuredMarkdownLoader, 
    UnstructuredODTLoader, 
    UnstructuredPowerPointLoader, 
    UnstructuredWordDocumentLoader, 
) 
 
 
from langchain.text_splitter import NLTKTextSplitter 
from langchain_community.vectorstores import Chroma 
from langchain.embeddings import HuggingFaceEmbeddings 
from langchain.docstore.document import Document 
 
 
import nltk 
nltk.download('punkt') 
nltk.download('punkt_tab') 
 
 
from constants import CHROMA_SETTINGS 
import chromadb 
 
 
# Load environment variables 
persist_directory = 'vectors' 
source_directory = os.environ.get('SOURCE_DIRECTORY', 'Contracts') 
embeddings_model_name = 'all-MiniLM-L12-v2' 
chunk_size = 500 
chunk_overlap = 90 
 
 
# Custom document loaders 
class MyElmLoader(UnstructuredEmailLoader): 
    """Wrapper to fallback to text/plain when default does not work""" 
 
 
    def load(self) -> List[Document]: 
        """Wrapper adding fallback for elm without html""" 
        try: 
            try: 
                doc = UnstructuredEmailLoader.load(self) 
            except ValueError as e: 
                if 'text/html content not found in email' in str(e): 
                    # Try plain text 
                    self.unstructured_kwargs["content_source"] = "text/plain" 
                    doc = UnstructuredEmailLoader.load(self) 
                else: 
                    raise 
        except Exception as e: 
            # Add file_path to exception message 
            raise type(e)(f"{self.file_path}: {e}") from e 
 
 
        return doc 
 
 
# Map file extensions to document loaders and their arguments 
LOADER_MAPPING = { 
    ".csv": (CSVLoader, {"encoding": "utf8"}), 
    ".doc": (UnstructuredWordDocumentLoader, {}), 
    ".docx": (UnstructuredWordDocumentLoader, {}), 
    ".enex": (EverNoteLoader, {}), 
    ".eml": (MyElmLoader, {}), 
    ".epub": (UnstructuredEPubLoader, {}), 
    ".html": (UnstructuredHTMLLoader, {}), 
    ".md": (UnstructuredMarkdownLoader, {}), 
    ".odt": (UnstructuredODTLoader, {}), 
    ".pdf": (PyPDFLoader, {}), 
    ".ppt": (UnstructuredPowerPointLoader, {}), 
    ".pptx": (UnstructuredPowerPointLoader, {}), 
    ".txt": (TextLoader, {"encoding": "utf8"}), 
} 
 
 
def load_single_document(file_path: str) -> List[Document]: 
    ext = "." + file_path.rsplit(".", 1)[-1].lower() 
    if ext in LOADER_MAPPING: 
        loader_class, loader_args = LOADER_MAPPING[ext] 
        loader = loader_class(file_path, **loader_args) 
        return loader.load() 
 
 
    raise ValueError(f"Unsupported file extension '{ext}'") 
     
def load_documents(source_dir: str, ignored_files: List[str] = []) -> List[Document]: 
    """ 
    Loads all documents from the source documents directory, ignoring specified files 
    """ 
    all_files = [] 
    for ext in LOADER_MAPPING: 
        all_files.extend( 
            glob.glob(os.path.join(source_dir, f"**/*{ext.lower()}"), recursive=True) 
        ) 
        all_files.extend( 
            glob.glob(os.path.join(source_dir, f"**/*{ext.upper()}"), recursive=True) 
        ) 
    filtered_files = list(set([file_path for file_path in all_files if file_path not in ignored_files])) 
    print(filtered_files) 
     
    results = []  # Initialize an empty list to store the loaded documents 
     
    # Create a progress bar 
    with tqdm(total=len(filtered_files), desc='Loading new documents', ncols=80) as pbar: 
        print("Starting Loading...") 
         
        for file_path in filtered_files: 
            # Load a single document using your load_single_document function 
            docs = load_single_document(file_path) 
             
            # Append the loaded documents to the results list 
            results.extend(docs) 
             
            # Update the progress bar 
            pbar.update() 
    return results 
 
 
def process_documents(ignored_files: List[str] = []) -> List[Document]: 
    """ 
    Load documents and split in chunks 
    """ 
    print(f"Loading documents from {source_directory}") 
    documents = load_documents(source_directory, ignored_files) 
    if not documents: 
        print("No new documents to load") 
        return "" 
    print(f"Loaded {len(documents)} new documents from {source_directory}") 
    text_splitter = NLTKTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap) 
    texts = text_splitter.split_documents(documents) 
    print(f"Split into {len(texts)} chunks of text (max. {chunk_size} tokens each)") 
    return texts 
 
 
def does_vectorstore_exist(persist_directory: str, embeddings: HuggingFaceEmbeddings) -> bool: 
    """ 
    Checks if vectorstore exists and if it has any documents 
    """ 
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS) 
    # Check if the vector store has any documents 
    if not db.get()['documents']: 
        return False 
    return True 
 
 
def main(): 
    # Create embeddings 
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name) 
    # Chroma client 
    chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory) 
 
 
    db = None  # Ensure `db` is always defined 
     
    # Check if vector store exists and set up db accordingly 
    if does_vectorstore_exist(persist_directory, embeddings): 
        # Reuse existing vector store with consistent settings 
        print(f"Appending to existing vectorstore at {persist_directory}") 
        db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client) 
        collection = db.get() 
        texts = process_documents([metadata['source'] for metadata in collection['metadatas']]) 
         
        if texts != "": 
            print(f"Creating embeddings. May take some minutes...") 
            db.add_documents(texts) 
    else: 
        # Create new vector store if it doesn't exist 
        print("Creating new vectorstore") 
        texts = process_documents() 
         
        if texts != "": 
            print(f"Creating embeddings. May take some minutes...") 
            db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS, client=chroma_client) 
 
 
    # Use the correct method to persist the vector store 
    if db: 
        db.persist()  # Ensure to use the correct method if it has it, or check documentation 
        db = None  # Make sure to clean up after use 
     
    print(f"DOCUMENT INGESTION COMPLETE!") 
 
 
if __name__ == "__main__": 
    main() 