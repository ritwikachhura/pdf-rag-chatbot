import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# from langchain.prompts import ChatPromptTemplate
# from langchain.schema.runnable import RunnablePassthrough
# from langchain.schema.output_parser import StrOutputParser

load_dotenv()

# Load the vector store you built in Step 2
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # top 4 most relevant chunks

# The prompt template — this is what makes it "grounded" and not just a chatbot,
# and it bakes in the "explain the document, don't give advice" boundary
template = """You are a document explainer. Answer the question using ONLY the context below,
which is an excerpt from the user's own document (rental agreement / insurance policy / loan terms).

Rules:
- Explain what the document says in plain, simple language.
- If the context doesn't contain the answer, say "This document doesn't seem to address that — you may want to check with the other party or a professional."
- Do NOT give legal or financial advice, recommendations, or opinions on what the user should do.
- Where relevant, quote the specific clause or section briefly so the user can verify it themselves.

Context:
{context}

Question: {question}

Answer:"""
prompt = ChatPromptTemplate.from_template(template)

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

if __name__ == "__main__":
    while True:
        question = input("\nAsk a question (or 'quit'): ")
        if question.lower() == "quit":
            break
        answer = rag_chain.invoke(question)
        print(f"\nAnswer: {answer}")
        
        print(retriever.invoke("{question}"))