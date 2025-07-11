# %%
import os
from sentence_transformers import SentenceTransformer
import torch.nn.functional as F
from tqdm import tqdm

# %%
class Sentence_embedding(object):
    def __init__(self, model_name, dim=768, device='cpu', prefix='') -> None:
        self.model_name = model_name
        self.prefix = prefix
        self.dim = dim  # only make sence when the model support dynamic dimision
        if self.model_name == 'nomic-v1':
            self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True, device=device)
        elif self.model_name == 'nomic-v1.5':
            self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True, device=device)
        else:
            raise ValueError(f'Unsupported model name: {self.model_name}')
        
        self.cache = {}
    
    def predict_batch(self, sentences: list) -> list:
        '''
        sentences: list of string
        Return: list of list of float
        '''
        if len(sentences) == 0:
            return []
        
        # add prefix
        prefix_sentences = []
        for sentence in sentences:
            prefix_sentences.append(self.prefix + sentence)

        if self.model_name == 'nomic-v1':
            embeddings = self.model.encode(prefix_sentences)
            embeddings = embeddings.tolist()
        elif self.model_name == 'nomic-v1.5':
            embeddings = self.model.encode(prefix_sentences, convert_to_tensor=True)
            embeddings = F.layer_norm(embeddings, normalized_shape=(embeddings.shape[1],))
            embeddings = embeddings[:, :self.dim]
            embeddings = F.normalize(embeddings, p=2, dim=1)
            embeddings = embeddings.tolist()
        else:
            raise ValueError(f'Unsupported model name: {self.model_name}')
        # cache
        for i, sentence in enumerate(sentences):
            self.cache[sentence] = embeddings[i]
        return embeddings
    
    def __call__(self, sentence: str) -> list:
        '''
        sentence: str
        Return: list of float
        '''
        # predict one sentence
        if sentence in self.cache:
            return self.cache[sentence]
        else:
            emb = self.predict_batch([sentence])[0]
            return emb
        
