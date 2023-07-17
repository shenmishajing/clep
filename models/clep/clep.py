from torch import nn
from torch.nn import functional as F


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

    def forward(self, data):
        x = self.token_embedding(data["signal"])
        x = x + self.wave_embedding(data["wave_embedding"])
        x = x + self.signal_embedding(data["signal_embedding"])[..., None, :]

        x = self.ecg_encoder(x.reshape(-1, *x.shape[-2:])).reshape(*x.shape)
        x = F.normalize(x, dim=-1)

        symbol_mask = data["symbol_target"] >= 0

        pred = x.matmul(
            other=F.normalize(self.symbol_embedding.weight.T, dim=-1)
        ).argmax(dim=-1)
        acc = (
            (
                (pred * symbol_mask == data["symbol_target"] * symbol_mask)
                * symbol_mask
            ).sum(dim=-1)
            / symbol_mask.sum(dim=-1)
        ).mean()

        data["symbol_target"][~symbol_mask] = 0
        symbol_target = F.normalize(
            self.symbol_embedding(data["symbol_target"]), dim=-1
        )

        symbol_loss = (
            1
            - (
                (symbol_mask * (x * symbol_target).sum(dim=-1).abs()).sum(dim=-1)
                / symbol_mask.sum(dim=-1)
            ).mean()
        )
        return {"loss": symbol_loss, "acc": acc}
