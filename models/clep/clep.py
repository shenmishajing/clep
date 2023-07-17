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
        symbol_kind_num=5,
        embedding_dim=512,
    ):
        super().__init__()

        self.token_size = token_size
        self.wave_kind_num = wave_kind_num
        self.signal_kind_num = signal_kind_num
        self.symbol_kind_num = symbol_kind_num
        self.embedding_dim = embedding_dim

        self.ecg_encoder = ecg_encoder
        self.token_embedding = nn.Linear(token_size, embedding_dim)
        self.wave_embedding = nn.Linear(wave_kind_num, embedding_dim)
        self.signal_embedding = nn.Embedding(signal_kind_num, embedding_dim)
        self.symbol_embedding = nn.Embedding(
            symbol_kind_num, wave_kind_num * embedding_dim
        )

        self.cls_tokens = nn.Parameter(torch.empty(wave_kind_num, embedding_dim))

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

        x = self.ecg_encoder(
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
        )[:, : self.wave_kind_num].reshape(*x.shape[:2], -1)

        x = F.normalize(x, dim=-1)

        pred = x.matmul(
            other=F.normalize(self.symbol_embedding.weight.T, dim=-1)
        ).argmax(dim=-1)
        acc = (pred == data["symbol_target"][:, None]).float().mean()

        symbol_target = F.normalize(
            self.symbol_embedding(data["symbol_target"]), dim=-1
        )[:, None]

        symbol_loss = 1 - (x * symbol_target).sum(dim=-1).mean()
        return {"loss": symbol_loss, "acc": acc}
