import pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import PromptTemplate, LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import SequentialChain


OPENAI_API_KEY = '<<Replace with your key>>'
PINECONE_API_KEY = '<<Replace with your key>>'
PINECONE_ENVIRONMENT = '<<Replace with your key>>'

# ---------------- Embedd FAQs ----------------

loader = TextLoader('faq.txt')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

print(docs[0].page_content)

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENVIRONMENT  # next to api key in console
)
index_name = "gym-faq"
docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)


with open('customer_email_group.txt') as f:
    email = f.read().replace('\n',' ')

# ---------------- Category Chain ----------------

llm = OpenAI(temperature=0,openai_api_key=OPENAI_API_KEY)

category_template = """
I want you to act as a professional email analyser and categorizer for Fitness Cartel Gym.
You have received below email from customer.
{email}
You need to categorize the email into one of Payments, Personal Training or Group Classes based on the content 
of the email. Your output need to be one of the above mentioned categories. No other text is needed.
Output:
"""
category_prompt_template = PromptTemplate(input_variables=["email"], template=category_template)
category_chain = LLMChain(llm=llm, prompt=category_prompt_template, output_key="category")

# ---------------- Summary Chain ----------------

summary_template = """
I want you to act as a professional email analyser for Fitness Cartel Gym.
You have received below email from customer.
{email}
You need to write a concise summary of the email in less than 20 words which explains the the intent.
. No other text is needed.
Output:
"""
summary_prompt_template = PromptTemplate(input_variables=["email"], template=summary_template)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt_template, output_key="summary")


# ---------------- Overall Chain ----------------

overall_chain = SequentialChain(
    chains=[category_chain, summary_chain],
    input_variables=["email"],
    # Here we return multiple variables
    output_variables=["category", "summary"],
    verbose=True)

result = overall_chain({"email":email})
print(result)
category = result["category"]
summary = result["summary"]


# ---------------- Similarity Chain ----------------

query = summary
docs = docsearch.similarity_search(query, k=2)
print(f"Found {len(docs)} documents.  Querying AI...")

chain = load_qa_chain(llm, chain_type="stuff")
result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
print(result["output_text"])
print()
best_match = result["output_text"]

# ---------------- Generate Email ----------------

email_gen_template = """
I want you to act as a customer support officer for a Fitness Cartel Gym.
You have received an email from customer asking for information about {email}.
You have already identified the question is about {category} and you have found the best answer as {answer}.
Draft an email as response with the above answer?
"""

email_gen_prompt_template = PromptTemplate.from_template(email_gen_template)
email_gen_chain = LLMChain(llm=llm, prompt=email_gen_prompt_template)

output = email_gen_chain.run({"email":email, "category":category, "answer":best_match})

print(output)










