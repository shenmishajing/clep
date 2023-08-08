from torch import nn

from .ecg_transformer import ECGTransformer


class ECGConvTransformer(ECGTransformer):
    def __init__(
        self,
        conv: nn.Module,
        token_embedding_enable=False,
        concat_embedding=False,
        *args,
        **kwargs
    ):
        super().__init__(
            *args,
            token_embedding_enable=token_embedding_enable,
            concat_embedding=concat_embedding,
            **kwargs
        )

        self.conv = conv

    def calculate_x(self, data):
        x = self.conv(data["signal"])
        return x
