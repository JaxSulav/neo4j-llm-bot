from transformers import AutoModel
from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings

# * Embedding Model
model = AutoModel.from_pretrained(
    "mixedbread-ai/mxbai-embed-large-v1", trust_remote_code=True
)

model_name = "mixedbread-ai/mxbai-embed-large-v1"
model_kwargs = {"device": "cpu"}
embeddings_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
)

# * Initializing LLM Ollama
llm = Ollama(model="mistral", temperature=0.4)
