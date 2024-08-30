import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np


class TransformerEmbedder:
    def __init__(self, model_name='nomic-ai/nomic-embed-text-v1.5'):
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)

    def generate_embeddings(self, texts, batch_size: int):
        embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = ["search_document: " + text for text in texts[i:i + batch_size]]
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Get the attention mask and convert it to float
                attention_mask = inputs['attention_mask'].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()

                # Mask the hidden states
                masked_hidden_states = outputs.last_hidden_state * attention_mask

                # Calculate the sum of the hidden states and the sum of the attention mask
                summed_hidden_states = masked_hidden_states.sum(dim=1)
                summed_mask = attention_mask.sum(dim=1)

                # Calculate the mean by dividing the summed hidden states by the summed mask
                batch_embeddings = (summed_hidden_states / summed_mask).cpu().numpy()
                embeddings.append(batch_embeddings)

        embeddings = np.concatenate(embeddings, axis=0)
        return embeddings