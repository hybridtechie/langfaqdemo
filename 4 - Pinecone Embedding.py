import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader

loader = TextLoader('faq.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

print(docs[0].page_content)

embeddings = OpenAIEmbeddings(openai_api_key='<<Replace with your key>>')
# initialize pinecone
pinecone.init(
    api_key='<<Replace with your key>>',  # find at app.pinecone.io
    environment='<<Replace with your key>>'  # next to api key in console
)

index_name = "gym-faq"

docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)

query = "Payments Failed"
docs = docsearch.similarity_search(query)

print(docs[0].page_content)







