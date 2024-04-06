from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv, find_dotenv

# Load environment variables from .env file
load_dotenv(find_dotenv(), override=True)
def ask_and_get_answer(vector_store, q, k=3):
    """
    Ask a question and get an answer using a vector store.

    Args:
        vector_store (Chroma): The Chroma vector store used for retrieval.
        q (str): The question to ask.
        k (int, optional): The number of answers to retrieve. Defaults to 3.

    Returns:
        str: The answer to the question.
    """
    # Import required modules
    from langchain.chains import RetrievalQA
    from langchain_openai import ChatOpenAI

    # Instantiate a language model (LLM) for conversation
    llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=1)

    # Create a retriever from the vector store for similarity search
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': k})

    # Create a retrieval-based QA chain
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    
    # Invoke the QA chain to get an answer to the question
    answer = chain.invoke(q)
    
    return answer

def chunk_data(data, chunk_size=256):
    """
    Split the input data into chunks.

    Args:
        data (str): The input data to be chunked.
        chunk_size (int, optional): The size of each chunk. Defaults to 256.

    Returns:
        list: A list of chunked data.
    """
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=20)
    
    # Split the data into chunks
    chunks = text_splitter.split_documents(data)
    
    return chunks

def create_embeddings_chroma(chunks, persist_directory='./chroma_db'):
    """
    Create Chroma embeddings from text chunks.

    Args:
        chunks (list): List of text chunks.
        persist_directory (str, optional): Directory to persist the vector store. Defaults to './chroma_db'.

    Returns:
        Chroma: The created Chroma vector store.
    """
    # Instantiate an embedding model from OpenAI (smaller version for efficiency)
    embeddings = OpenAIEmbeddings(model='text-embedding-3-small', dimensions=1536)  
    # Create a Chroma vector store using the provided text chunks and embedding model, 
    # configuring it to save data to the specified directory 
    vector_store = Chroma.from_documents(chunks, embeddings, persist_directory=persist_directory) 

    return vector_store  # Return the created vector store

def load_data(path):
    """
    Load data from PDF files, chunk them, and create Chroma embeddings.

    Args:
        path (str): Path to the directory containing PDF files.

    Returns:
        Chroma: The created Chroma vector store.
    """
    # Load PDF documents from the specified directory
    loader = PyPDFDirectoryLoader(path)
    docs = loader.load()
    
    # Split the documents into chunks
    chunks = chunk_data(docs, chunk_size=256)
    
    # Create Chroma embeddings from the chunks
    vector_store = create_embeddings_chroma(chunks)

if __name__ == "__main__":
    path = "data/"
    load_data(path)
    '''
    # Asking questions
    q = 'What is Vertex AI Search?'
    answer = ask_and_get_answer(vector_store, q)
    print(answer)
    '''
    