import torch
from transformers.cache_utils import DynamicCache
from typing import Optional, List, Tuple, Dict, Any

class DiffusionDynamicCache(DynamicCache):
    def __init__(self, num_hidden_layers: Optional[int] = None):
        super().__init__(num_hidden_layers)

    def full_update(
        self,
        new_kv: Tuple,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ):
        for i, (key, val) in enumerate(new_kv):
            self.layers[i].keys = torch.cat([self.layers[i].keys, key], dim=-2)
            self.layers[i].values = torch.cat([self.layers[i].values, val], dim=-2)
    
    def select_partial(
        self,
        indices: torch.Tensor,
    ):
        for i in range(len(self.layers)):
            self.layers[i].keys = self.layers[i].keys[:, :, indices, :]
            self.layers[i].values = self.layers[i].values[:, :, indices, :]
    
    def batch_select_minibatch(self, indices: torch.Tensor):
        """Only keep the `indices` in the batch dimension of the cache. Used in contrastive search."""
        for layer_idx in range(len(self)):
            self.layers[layer_idx].keys = self.layers[layer_idx].keys[:indices, ...]
            self.layers[layer_idx].values = self.layers[layer_idx].values[:indices, ...]
