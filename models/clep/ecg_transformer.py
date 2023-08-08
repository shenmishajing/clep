import math

import torch
from torch import nn
from torch.nn.init import xavier_uniform_


class ECGTransformer(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        token_size=1,
        wave_kind_num=3,
        signal_kind_num=5,
        wave_num_cls_token=True,
        wave_attention=True,
        embedding_dim=32,
        token_embedding_enable=True,
        wave_embedding_enable=True,
        signal_embedding_enable=True,
        pos_embedding_enable=True,
        concat_embedding=True,
        fc_per_cls_token=False,
        num_classes=5,
        multi_label=False,
    ):
        super().__init__()

        self.token_size = token_size
        self.wave_kind_num = wave_kind_num
        self.signal_kind_num = signal_kind_num
        self.wave_num_cls_token = wave_num_cls_token
        self.cls_token_num = wave_kind_num if wave_num_cls_token else 1
        self.wave_attention = wave_attention
        self.embedding_dim = embedding_dim
        self.token_embedding_enable = token_embedding_enable
        self.wave_embedding_enable = wave_embedding_enable
        self.signal_embedding_enable = signal_embedding_enable
        self.pos_embedding_enable = pos_embedding_enable
        self.concat_embedding = concat_embedding
        self.fc_per_cls_token = fc_per_cls_token
        self.multi_label = multi_label
        self.num_classes = num_classes

        self.encoder = encoder

        embedding_dim_factor = 1

        if token_embedding_enable:
            self.token_embedding = nn.Linear(token_size, embedding_dim)

        if wave_embedding_enable:
            self.wave_embedding = nn.Linear(wave_kind_num, embedding_dim)
            if concat_embedding:
                embedding_dim_factor += 1

        if signal_embedding_enable:
            self.signal_embedding = nn.Embedding(signal_kind_num, embedding_dim)
            if concat_embedding:
                embedding_dim_factor += 1

        if pos_embedding_enable and concat_embedding:
            embedding_dim_factor += 1

        if not fc_per_cls_token:
            embedding_dim_factor *= self.cls_token_num

        self.embedding_dim_factor = embedding_dim_factor

        self.cls_tokens = nn.Parameter(torch.empty(self.cls_token_num, embedding_dim))

        self.fc = nn.Linear(embedding_dim_factor * embedding_dim, num_classes)

        if multi_label:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.cls_tokens)

    def embedding(
        self,
        x,
        attention_mask,
        wave_embedding=None,
        signal_embedding=None,
        peak_position=None,
        *args,
        **kwargs
    ):
        if self.token_embedding_enable:
            x = self.token_embedding(x)

        x = torch.cat(
            [self.cls_tokens[None, None].expand(*x.shape[:2], -1, -1), x], dim=-2
        )

        x_with_embedding = [x]

        attention_mask = (
            attention_mask.reshape(*attention_mask.shape[:1], -1, self.token_size)
            .min(-1)[0]
            .to(torch.bool)
        )
        attention_mask = torch.cat(
            [
                attention_mask.new_zeros((attention_mask.shape[0], self.cls_token_num)),
                attention_mask,
            ],
            -1,
        )
        attention_mask = attention_mask[:, None].repeat(
            (1, attention_mask.shape[-1], 1)
        )

        wave_embedding = wave_embedding.reshape(
            *wave_embedding.shape[:1], -1, self.token_size, self.wave_kind_num
        ).mean(-2)

        if self.wave_attention:
            attention_mask[:, : self.cls_token_num] = True
            for i in range(self.cls_token_num):
                if self.wave_num_cls_token:
                    wave_kind = wave_embedding[:, :, i]
                else:
                    wave_kind = wave_embedding.sum(-1)
                attention_mask[:, i, i] = False
                attention_mask[:, i, self.cls_token_num :] = wave_kind == 0

        if self.wave_embedding_enable:
            if self.wave_num_cls_token:
                wave_embedding = torch.cat(
                    [
                        torch.eye(
                            self.cls_token_num,
                            dtype=wave_embedding.dtype,
                            device=wave_embedding.device,
                        )[None].expand(*wave_embedding.shape[:1], -1, -1),
                        wave_embedding,
                    ],
                    dim=-2,
                )
            else:
                wave_embedding = torch.cat(
                    [
                        wave_embedding.new_ones(
                            [
                                wave_embedding.shape[0],
                                self.cls_token_num,
                                wave_embedding.shape[-1],
                            ]
                        ),
                        wave_embedding,
                    ],
                    dim=-2,
                )
            wave_embedding = self.wave_embedding(wave_embedding)[:, None].expand_as(x)
            x_with_embedding.append(wave_embedding)

        if self.signal_embedding_enable:
            signal_embedding = self.signal_embedding(signal_embedding)[
                ..., None, :
            ].expand_as(x)
            x_with_embedding.append(signal_embedding)

        if self.pos_embedding_enable:
            pos_embedding = x.new_zeros([x.shape[0], *x.shape[-2:]])

            position = torch.arange(0, x.shape[-2], device=x.device)
            if peak_position is None:
                peak_position = position.new_zeros((x.shape[0],))
            position = position[None, :, None] - peak_position[:, None, None]
            div_term = torch.exp(
                torch.arange(0, x.shape[-1], 2, device=x.device)
                * -(math.log(10000.0) / x.shape[-1])
            )[None, None]
            pos_embedding[..., 0::2] = torch.sin(position * div_term)  # 偶数下标的位置
            pos_embedding[..., 1::2] = torch.cos(position * div_term)  # 奇数下标的位置
            pos_embedding = pos_embedding[:, None].expand_as(x)
            x_with_embedding.append(pos_embedding)

        if len(x_with_embedding) > 1:
            if self.concat_embedding:
                x = torch.cat(x_with_embedding, dim=-1)
            else:
                x = sum(x_with_embedding)

        return x, attention_mask

    def calculate_x(self, data):
        x = data["signal"]
        x = x.reshape(*x.shape[:2], -1, self.token_size)
        return x

    def encoder_forward(self, x, attention_mask):
        return self.encoder(
            x.reshape(-1, *x.shape[-2:]),
            attention_mask[:, None, None, ...]
            .expand(-1, x.shape[1], self.encoder.layers[0].self_attn.num_heads, -1, -1)
            .reshape(-1, *attention_mask.shape[-2:]),
        )

    def extract_feat(self, x, attention_mask):
        x = self.transformer_forward(x, attention_mask)[
            :, : self.cls_token_num
        ].reshape(*x.shape[:2], -1)
        return x

    def calculate_pred(self, x, data):
        pred = self.fc(x).mT
        return pred

    def calculate_loss(self, pred, target, x):
        target = target[..., None]
        if self.multi_label:
            target = target.expand(-1, -1, x.shape[1])
            loss = self.loss(pred.sigmoid(), target.to(pred.dtype))
        else:
            target = target.expand(-1, x.shape[1])
            loss = self.loss(pred, target)
        return loss, target

    def forward(self, data):
        x = self.calculate_x(data)
        x, attention_mask = self.embedding(x=x, **data)
        x = self.extract_feat(x, attention_mask)

        pred = self.calculate_pred(x, data)
        loss, target = self.calculate_loss(pred, data["target"], x)

        return {"log_dict": {"loss": loss}, "pred": pred, "target": target}
