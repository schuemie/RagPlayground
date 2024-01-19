from transformers import AutoModel
from numpy.linalg import norm
from langchain.embeddings import HuggingFaceEmbeddings

cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method
embeddings = model.encode(['What is the capital of Argentina?', 'Amsterdam is not the capital of Argentina.'])
print(cos_sim(embeddings[0], embeddings[1]))

embeddings = model.encode(['What is the capital of Argentina?', 'Buenos Aires is the capital of Argentina.'])
print(cos_sim(embeddings[0], embeddings[1]))