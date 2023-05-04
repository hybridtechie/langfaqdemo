from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
import os
# import nltk

# nltk.download('punkt')
loader = UnstructuredFileLoader("./faqs/faq.txt")
docs = loader.load()

print(docs[0].page_content[:20])

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

print(f"{len(texts)} texts created")

db = Chroma.from_documents(documents=texts, embedding=OpenAIEmbeddings(), persist_directory="data")
db.persist()