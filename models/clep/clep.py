from torch import nn


class CLEP(nn.Module):
    def __init__(
        self,
        ecg_encoder: nn.Module,
        token_size: int,
        wave_kind_num=3,
        singla_kind_num=5,
        embedding_dim=512,
    ):
        super().__init__()

        self.ecg_encoder = ecg_encoder
        self.token_embedding = nn.Linear(token_size, embedding_dim)
        self.wave_embedding = nn.Linear(wave_kind_num, embedding_dim)
        self.signal_embedding = nn.Embedding(singla_kind_num, embedding_dim)

    def forward(self, x):
        signal = self.token_embedding(x["signal"])
        signal = signal + self.wave_embedding(x["wave_embedding"])
        signal = signal + self.signal_embedding(x["signal_embedding"])[..., None, :]

        signal = self.ecg_encoder(signal.reshape(-1, *signal.shape[-2:])).reshape(
            *signal.shape
        )
        return x
