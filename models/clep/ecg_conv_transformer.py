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
        x = data["signal"]
        batch_size, single_num = x.shape[:2]
        x = x.reshape(batch_size * single_num, -1, self.conv.conv1.in_channels).mT

        x = self.conv.extract_feat(x)

        return x.reshape(batch_size, single_num, *x.shape[-2:]).mT


class ECGConvTransformerWithChannel(ECGConvTransformer):
    def calculate_x(self, data):
        x = data["signal"]
        x = x.reshape(*x.shape[:-2], -1)
        return self.conv.extract_feat(x)[:, None].mT
