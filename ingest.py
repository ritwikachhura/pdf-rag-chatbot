import os
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import time

load_dotenv(override=True)


# 1. Load all PDFs from the docs folder
loader = DirectoryLoader("docs/", glob="**/*.pdf", loader_cls=PyPDFLoader) # type: ignore
documents = loader.load()
print(f"Loaded {len(documents)} pages")

# 2. Split into chunks — 1000 characters with 200 overlap is a solid default
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# 3. Embed each chunk and store in a local vector database (Chroma)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

batch_size = 50

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    vectorstore.add_documents(batch)

    print(
        f"Processed batch {i // batch_size + 1}/"
        f"{(len(chunks) + batch_size - 1)//batch_size}"
    )
    time.sleep(5)

print("Vector store built successfully")