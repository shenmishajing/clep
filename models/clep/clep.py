import pickle

import torch
from torch import nn
from torch.nn import functional as F

from .ecg_transformer import ECGTransformer


class CLEP(ECGTransformer):
    def __init__(
        self,
        *args,
        symbol_embedding_dim=1536,
        symbol_embedding_path=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.symbol_embedding = nn.ParameterDict(
            pickle.load(open(symbol_embedding_path, "rb"))
        )
        for p in self.symbol_embedding.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(4 * self.embedding_dim, symbol_embedding_dim)

    def forward(self, data):
        x = self.fc(self.transformer_forward(data))

        symbol_embedding = []
        for signals in data["signal_name"]:
            symbol_embedding.append(
                torch.stack([self.symbol_embedding[signal] for signal in signals])
            )
        symbol_embedding = torch.stack(symbol_embedding)

        if self.normalize_loss:
            x = F.normalize(x, dim=-1)
            symbol_embedding = F.normalize(symbol_embedding, dim=-1)

        pred = (symbol_embedding.matmul(x[..., None]).squeeze(-1).mT + 1) / 2

        target = data["symbol_target"][..., None]
        if self.multi_label:
            target = target.expand(-1, -1, x.shape[1])
            loss = self.loss(pred, target.to(pred.dtype))
        else:
            target = target.expand(-1, x.shape[1])
            loss = self.loss(pred, target)

        return {"log_dict": {"loss": loss}, "pred": pred, "target": target}
