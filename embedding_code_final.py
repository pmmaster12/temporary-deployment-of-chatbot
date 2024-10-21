from langchain_community.document_loaders import SeleniumURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.text_splitter import NLTKTextSplitter
import hashlib
from langchain_redis import RedisVectorStore
from langchain_community.embeddings import OllamaEmbeddings

def generate_hash(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def is_hash_in_file(link_path, output_file_path):
    try:
        with open(output_file_path, 'r') as file:
            lines = file.readlines()
            return link_path + '\n' in lines
    except FileNotFoundError:
        return False

def embedding(urls):
    path = "hashval1.txt"
    path1 = "existed_empty_invalid_url1.txt"
    path2 = "valid_url1.txt"
    
    try:
        # Check if the URL is empty or invalid
        if not urls.strip():
            raise ValueError("Empty URL provided")

        loader = SeleniumURLLoader([urls])
        
        # Try loading the URL content
        data = loader.load()
        
        # If the data is empty or invalid, raise an error
        if not data or not data[0].page_content:
            raise ValueError("Invalid or empty content from URL")

        # Generate a hash value for the content
        hash_val = generate_hash(''.join([str(doc.page_content) for doc in data]))

        # Check if the hash already exists
        if not is_hash_in_file(hash_val, path):
            # Split the document into chunks
            splitter = NLTKTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)
            chunks = splitter.split_documents(data)

            # Store the chunks in Redis vector store
            vector_db_new = RedisVectorStore.from_documents(
                documents=chunks,
                embedding=OllamaEmbeddings(model="nomic-embed-text", show_progress=True),
                index_name='new'
            )

            # Save hash and valid URL
            with open(path, 'a') as file:
                file.write(f"{hash_val}\n")
            with open(path2, 'a') as file:
                file.write(f"{urls}\n")
        else:
            print(f"Already existed URL - {urls}")
            with open(path1, 'a') as file:
                file.write(f"{urls}\n")
    
    except Exception as e:
        print(f"Error processing {urls}: {e}")
        with open(path1, 'a') as file:
            file.write(f"Failed to process {urls} - Error: {e}\n")

# Read URLs from the file and process them
with open('testing_scrap_url1.txt', 'r') as file:
    url = [str(line.strip()) for line in file.readlines()]
    for i in url:
        if i:
            embedding(i)
