import pickle

import torch
from torch import nn
from torch.nn import functional as F

from .ecg_transformer import ECGTransformer


class CLEP(ECGTransformer):
    def __init__(
        self,
        *args,
        waves="PRT",
        symbols="LRVA",
        normalize_loss: bool = True,
        symbol_embedding_dim=1536,
        symbol_embedding_path=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.waves = waves
        self.symbols = symbols
        self.normalize_loss = normalize_loss

        self.symbol_embedding = pickle.load(open(symbol_embedding_path, "rb"))
        self._symbol_embedding = nn.ParameterList()
        for lead in self.symbol_embedding:
            for disease in self.symbol_embedding[lead]:
                for wave in self.symbol_embedding[lead][disease]:
                    self.symbol_embedding[lead][disease][wave] = nn.Parameter(
                        self.symbol_embedding[lead][disease][wave]
                    )
                    self._symbol_embedding.append(
                        self.symbol_embedding[lead][disease][wave]
                    )
        for p in self._symbol_embedding.parameters():
            p.requires_grad = False

        self.fc = nn.Linear(4 * self.embedding_dim, symbol_embedding_dim)

        if normalize_loss:
            self.t = nn.Parameter(torch.zeros(()))
        else:
            self.t = None

    def forward(self, data):
        # b, c, l, d: batch_size, lead_num, seq_len, embedding_dim
        x = self.embedding(data)

        # b, c, w, d: batch_size, lead_num, wave_kind_num, embedding_dim
        x = self.fc(
            # b * c, w, d: batch_size * lead_num, wave_kind_num, embedding_dim
            self.transformer_forward(x, data["attention_mask"])[:, : self.wave_kind_num]
        ).reshape(*x.shape[:2], len(self.waves), -1)

        pred = []
        for batch_ind, leads in enumerate(data["signal_name"]):
            cur_pred = []
            for lead_ind, lead in enumerate(leads):
                cur_p = []
                for symbol in self.symbols:
                    cur_x = []
                    symbol_embedding = []
                    for i, wave in enumerate(self.waves):
                        if wave in self.symbol_embedding[lead][symbol]:
                            # d: embedding_dim
                            cur_x.append(x[batch_ind, lead_ind, i, :])
                            # d: embedding_dim
                            symbol_embedding.append(
                                self.symbol_embedding[lead][symbol][wave]
                            )

                    # w * d: wave_kind_num * embedding_dim
                    cur_x = torch.cat(cur_x, dim=-1)
                    # w * d: wave_kind_num * embedding_dim
                    symbol_embedding = torch.cat(symbol_embedding, dim=-1)

                    if self.normalize_loss:
                        cur_x = F.normalize(cur_x, dim=-1)
                        symbol_embedding = F.normalize(symbol_embedding, dim=-1)

                    # 1: 1
                    cur_p.append((cur_x * symbol_embedding).sum())

                # s: symbol_num
                cur_pred.append(torch.stack(cur_p))

            # s, c: symbol_num, lead_num
            pred.append(torch.stack(cur_pred, dim=-1))

        # b, s, c: batch_size, symbol_num, lead_num
        pred = torch.stack(pred, dim=0)

        target = data["symbol_target"][..., None]
        if self.multi_label:
            target = target.expand(-1, -1, x.shape[1])
            loss = self.loss(pred.sigmoid(), target.to(pred.dtype))
        else:
            target = target.expand(-1, x.shape[1])
            loss = self.loss(pred, target)

        return {"log_dict": {"loss": loss}, "pred": pred, "target": target}
