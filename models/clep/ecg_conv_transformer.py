import torch
from torch import nn
from torch.nn.init import xavier_uniform_


class ECGConvTransformer(nn.Module):
    def __init__(
        self,
        conv_encoder: nn.Module,
        transformer_encoder: nn.Module,
        token_size: int,
        wave_kind_num=3,
        signal_kind_num=5,
        wave_num_cls_token=True,
        wave_attention=True,
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
        self.wave_attention = wave_attention
        self.multi_label = multi_label
        self.num_classes = num_classes

        self.conv_encoder = conv_encoder
        self.transformer_encoder = transformer_encoder
        # self.wave_embedding = nn.Linear(wave_kind_num, embedding_dim)
        # self.signal_embedding = nn.Embedding(signal_kind_num, embedding_dim)

        self.cls_tokens = nn.Parameter(torch.empty(self.cls_token_num, embedding_dim))

        self.fc = nn.Linear(self.cls_token_num * embedding_dim, num_classes)

        if multi_label:
            self.loss = nn.BCELoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.cls_tokens)

    def embedding(self, data, x):
        x = torch.cat(
            [self.cls_tokens[None, None].expand(*x.shape[:2], -1, -1), x], dim=-2
        )

        attention_mask = data["attention_mask"]
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

        if self.wave_attention:
            wave_embedding = data["wave_embedding"]
            wave_embedding = wave_embedding.reshape(
                *wave_embedding.shape[:1], -1, self.token_size, self.wave_kind_num
            ).mean(-2)
            attention_mask[:, : self.cls_token_num] = True
            for i in range(self.cls_token_num):
                if self.wave_num_cls_token:
                    wave_kind = wave_embedding[:, :, i]
                else:
                    wave_kind = wave_embedding.sum(-1)
                attention_mask[:, i, i] = False
                attention_mask[:, i, self.cls_token_num :] = wave_kind == 0

        # if self.wave_num_cls_token:
        #     wave_embedding = torch.cat(
        #         [
        #             torch.eye(
        #                 self.cls_token_num,
        #                 dtype=wave_embedding.dtype,
        #                 device=wave_embedding.device,
        #             )[None].expand(*wave_embedding.shape[:1], -1, -1),
        #             wave_embedding,
        #         ],
        #         dim=-2,
        #     )
        # else:
        #     wave_embedding = torch.cat(
        #         [
        #             wave_embedding.new_ones(
        #                 [
        #                     wave_embedding.shape[0],
        #                     self.cls_token_num,
        #                     wave_embedding.shape[-1],
        #                 ]
        #             ),
        #             wave_embedding,
        #         ],
        #         dim=-2,
        #     )
        # wave_embedding = self.wave_embedding(wave_embedding)[:, None].expand_as(x)
        # signal_embedding = self.signal_embedding(data["signal_embedding"])[
        #     ..., None, :
        # ].expand_as(x)

        # pos_embedding = x.new_zeros([x.shape[0], *x.shape[-2:]])

        # position = torch.arange(0, x.shape[-2], device=x.device)
        # position = (
        #     position[None, :, None]
        #     - data["peak_position"][:, None, None] / self.token_size
        # )
        # div_term = torch.exp(
        #     torch.arange(0, x.shape[-1], 2, device=x.device)
        #     * -(math.log(10000.0) / x.shape[-1])
        # )[None, None]
        # pos_embedding[..., 0::2] = torch.sin(position * div_term)  # 偶数下标的位置
        # pos_embedding[..., 1::2] = torch.cos(position * div_term)  # 奇数下标的位置
        # pos_embedding = pos_embedding[:, None].expand_as(x)

        # x = torch.cat([x, wave_embedding, signal_embedding, pos_embedding], dim=-1)

        return x, attention_mask

    def transformer_forward(self, x, attention_mask):
        return self.transformer_encoder(
            x.reshape(-1, *x.shape[-2:]),
            attention_mask[:, None, None, ...]
            .expand(
                -1,
                x.shape[1],
                self.transformer_encoder.layers[0].self_attn.num_heads,
                -1,
                -1,
            )
            .reshape(-1, *attention_mask.shape[-2:]),
        )

    def forward(self, data):
        x = self.conv_encoder(data["signal"])
        x, attention_mask = self.embedding(data, x)
        x = self.transformer_forward(x, attention_mask)[
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
