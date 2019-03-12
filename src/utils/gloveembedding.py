import torch
import torch.nn as nn
from torch.nn import functional as F

from word_embedding import Glove
from log import info

class GloveEmbedding(nn.Module):
    """
    A module for loading and embedding into pretrained GloVe
    """
    def __init__(self, glove: Glove):
        """
        :param glove: Glove
        """
        super().__init__()
        # info
        info("Create GloveEmbedding w/ embed_dim: {e_dim}"
             .format(e_dim=glove.embedding_dim))
        self.glove = glove
        self.num_special_tokens = self.glove.token_mapper.mapped_output_size()
        self.embedding_dim = self.glove.embedding_dim

        # Special embeddings use the last index as padding
        self._special_embeddings = nn.Embedding(
            self.num_special_tokens + 1,
            self.embedding_dim,
            padding_idx=self.num_special_tokens
        )
        # Pretrained embeddings uses the 0 index as padding
        self._pretrained_weights = torch.from_numpy(self.glove.numbers)
        self.reset_parameters()

    def reset_parameters(self):
        if self.num_special_tokens > 0:
            nn.init.normal_(self._special_embeddings.weight[:self.num_special_tokens], 0.0, 0.1)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        # Embed special tokens
        special_embeddings_indices = torch.clamp(indices, 0, self.num_special_tokens)
        special_embed = self._special_embeddings(special_embeddings_indices)

        # Embed pretrained tokens
        pretrained_embeddings_indices = torch.clamp(
            indices - (self.num_special_tokens - 1),
            0,
            self.glove.vocab_size + 1
        )
        # Embed pretrained embeddings by manually calling backend (based on internal embedding code in pytorch 0.3.1)
        pretrained_embed = F.embedding(pretrained_embeddings_indices, self._pretrained_weights)

        # Add the two to produce the final embedded vector
        output = special_embed + pretrained_embed

        return output

    def __call__(self, indices: torch.Tensor):
        return super().__call__(indices)

    def cuda(self, device=None):
        raise ValueError("Glove embeddings should always live on the CPU")

