# Desafio MBA Engenharia de Software com IA - Full Cycle

Para execução do chat será necessário realizar os seguintes passos

##Preparação do Ambiente

`python3 -m venv venv`
`source venv/bin/activate`

### Variáveis de Ambiente

Crie o arquivo `.env` com as seguintes informações

````
# Google Gemini API key
GOOGLE_API_KEY=<informe sua API Key do Google>
GOOGLE_EMBEDDING_MODEL=gemini-embedding-001
PGVECTOR_URL=postgresql+psycopg://postgres:postgres@localhost:5432/rag
PGVECTOR_COLLECTION=gpt5_collection
PDF_PATH=document.pdf

````

##Iniciar Postgres com PGVector 

`docker compose up -d`

##Executando a ingestão 

`python3 src/ingest.py `

##Executando o chat

`python3 src/chat.py`

Para realizar a saída do chat basta informar a palavra exit