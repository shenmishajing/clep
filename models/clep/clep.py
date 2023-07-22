import pickle

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_


class CLEP(nn.Module):
    def __init__(
        self,
        ecg_encoder: nn.Module,
        token_size: int,
        wave_kind_num=3,
        signal_kind_num=5,
        embedding_dim=512,
        symbol_embedding_dim=1536,
        symbol_embedding_path=None,
        symbol_loss: nn.Module = nn.CrossEntropyLoss(),
        normalize_loss: bool = True,
    ):
        super().__init__()

        self.token_size = token_size
        self.wave_kind_num = wave_kind_num
        self.signal_kind_num = signal_kind_num
        self.embedding_dim = embedding_dim

        self.ecg_encoder = ecg_encoder
        self.token_embedding = nn.Linear(token_size, embedding_dim)
        self.wave_embedding = nn.Linear(wave_kind_num, embedding_dim)
        self.signal_embedding = nn.Embedding(signal_kind_num, embedding_dim)

        self.cls_tokens = nn.Parameter(torch.empty(wave_kind_num, embedding_dim))

        self.symbol_embedding = nn.ParameterDict(
            pickle.load(open(symbol_embedding_path, "rb"))
        )
        for p in self.symbol_embedding.parameters():
            p.requires_grad = False

        self.symbol_fc = nn.Linear(embedding_dim, symbol_embedding_dim)

        self.symbol_loss = symbol_loss
        self.normalize_loss = normalize_loss

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.cls_tokens)

    def forward(self, data):
        x = self.token_embedding(data["signal"])
        x = torch.cat(
            [self.cls_tokens[None, None].expand(*x.shape[:2], -1, -1), x], dim=-2
        )
        x = x + self.wave_embedding(data["wave_embedding"])[:, None]
        x = x + self.signal_embedding(data["signal_embedding"])[..., None, :]

        x = self.symbol_fc(
            self.ecg_encoder(
                x.reshape(-1, *x.shape[-2:]),
                data["attention_mask"][:, None, None, ...]
                .expand(
                    -1,
                    x.shape[1],
                    self.ecg_encoder.layers[0].self_attn.num_heads,
                    -1,
                    -1,
                )
                .reshape(-1, *data["attention_mask"].shape[-2:]),
            )[:, : self.wave_kind_num]
        ).reshape(*x.shape[:2], -1)

        symbol_embedding = []
        for signals in data["signal_name"]:
            symbol_embedding.append(
                torch.stack([self.symbol_embedding[signal] for signal in signals])
            )
        symbol_embedding = torch.stack(symbol_embedding)

        if self.normalize_loss:
            x = F.normalize(x, dim=-1)
            symbol_embedding = F.normalize(symbol_embedding, dim=-1)

        target = data["symbol_target"][:, None].expand(-1, x.shape[1])

        pred = symbol_embedding.matmul(x[..., None]).squeeze(-1).mT
        if isinstance(self.symbol_loss, nn.NLLLoss):
            pred = pred + 1
            pred = pred / pred.sum(dim=1, keepdim=True)
            loss = self.symbol_loss(pred, target) + 1
        else:
            loss = self.symbol_loss(pred, target)
            pred = pred.sigmoid()

        return {"log_dict": {"loss": loss}, "pred": pred, "target": target}
