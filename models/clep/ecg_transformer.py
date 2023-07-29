import math

import torch
from torch import nn
from torch.nn.init import xavier_uniform_


class ECGTransformer(nn.Module):
    def __init__(
        self,
        ecg_encoder: nn.Module,
        token_size: int,
        wave_kind_num=3,
        signal_kind_num=5,
        wave_num_cls_token=True,
        embedding_dim=32,
        num_classes=5,
        multi_label=False,
    ):
        super().__init__()

        self.token_size = token_size
        self.wave_kind_num = wave_kind_num
        self.signal_kind_num = signal_kind_num
        self.wave_num_cls_token = wave_num_cls_token
        self.cls_token_num = wave_kind_num if wave_num_cls_token else 1
        self.embedding_dim = embedding_dim
        self.multi_label = multi_label
        self.num_classes = num_classes

        self.ecg_encoder = ecg_encoder
        self.token_embedding = nn.Linear(token_size, embedding_dim)
        self.wave_embedding = nn.Linear(wave_kind_num, embedding_dim)
        self.signal_embedding = nn.Embedding(signal_kind_num, embedding_dim)

        self.cls_tokens = nn.Parameter(torch.empty(self.cls_token_num, embedding_dim))

        self.fc = nn.Linear(4 * self.cls_token_num * embedding_dim, num_classes)

        if multi_label:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.cls_tokens)

    def embedding(self, data):
        x = self.token_embedding(data["signal"])
        x = torch.cat(
            [self.cls_tokens[None, None].expand(*x.shape[:2], -1, -1), x], dim=-2
        )
        wave_embedding = self.wave_embedding(data["wave_embedding"])[:, None].expand_as(
            x
        )
        signal_embedding = self.signal_embedding(data["signal_embedding"])[
            ..., None, :
        ].expand_as(x)

        pos_embedding = x.new_zeros([x.shape[0], *x.shape[-2:]])
        position = data["position_embedding"][..., None]
        div_term = torch.exp(
            torch.arange(0, x.shape[-1], 2, device=x.device)
            * -(math.log(10000.0) / x.shape[-1])
        )[None, None]
        pos_embedding[..., 0::2] = torch.sin(position * div_term)  # 偶数下标的位置
        pos_embedding[..., 1::2] = torch.cos(position * div_term)  # 奇数下标的位置
        pos_embedding = pos_embedding[:, None].expand_as(x)

        x = torch.cat([x, wave_embedding, signal_embedding, pos_embedding], dim=-1)

        return x

    def transformer_forward(self, x, attention_mask):
        return self.ecg_encoder(
            x.reshape(-1, *x.shape[-2:]),
            attention_mask[:, None, None, ...]
            .expand(
                -1, x.shape[1], self.ecg_encoder.layers[0].self_attn.num_heads, -1, -1
            )
            .reshape(-1, *attention_mask.shape[-2:]),
        )

    def forward(self, data):
        x = self.embedding(data)
        x = self.transformer_forward(x, data["attention_mask"])[
            :, : self.cls_token_num
        ].reshape(*x.shape[:2], -1)

        pred = self.fc(x).mT

        target = data["target"][..., None]
        if self.multi_label:
            target = target.expand(-1, -1, x.shape[1])
            loss = self.loss(pred.sigmoid(), target.to(pred.dtype))
        else:
            target = target.expand(-1, x.shape[1])
            loss = self.loss(pred, target)

        return {"log_dict": {"loss": loss}, "pred": pred, "target": target}
