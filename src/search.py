import os 
import sys
from pathlib import Path
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from langchain_postgres import PGVector

load_dotenv()

for k in ("GOOGLE_API_KEY", "GOOGLE_EMBEDDING_MODEL", "PGVECTOR_URL", "PGVECTOR_COLLECTION"):
    if not os.getenv(k):
        raise RuntimeError(f"environment variable {k} is not set")


embeddings = GoogleGenerativeAIEmbeddings(model= os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001"))

store = PGVector(
    embeddings=embeddings, 
    collection_name=os.getenv("PGVECTOR_COLLECTION"),
    connection=os.getenv("PGVECTOR_URL"), 
    use_jsonb=True
)

prompt = PromptTemplate.from_template(
  """
    CONTEXTO:
    {contexto}

    REGRAS:
    - Responda somente com base no CONTEXTO.
    - Se a informação não estiver explicitamente no CONTEXTO, responda:
      "Não tenho informações necessárias para responder sua pergunta."
    - Nunca invente ou use conhecimento externo.
    - Nunca produza opiniões ou interpretações além do que está escrito.

    EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
    Pergunta: "Qual é a capital da França?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    Pergunta: "Quantos clientes temos em 2024?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    Pergunta: "Você acha isso bom ou ruim?"
    Resposta: "Não tenho informações necessárias para responder sua pergunta."

    PERGUNTA DO USUÁRIO:
    {pergunta}

    RESPONDA A "PERGUNTA DO USUÁRIO"
  """
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.5)

chain = prompt | llm

def search_prompt(question=None):
    while True:
      conversation = input("Pergunta (or 'exit' to quit): ")    
      if conversation.lower() == 'exit':        
          sys.exit() # This breaks out of the loop, ending the program

      result = store.similarity_search_with_score(conversation, k=10)

      response = chain.invoke({
        "contexto" : result,
        "pergunta" : conversation
        })
      print("Resposta: ", response.content)
      print("-"*30)

