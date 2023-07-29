import pickle

import torch
from torch import nn
from torch.nn import functional as F

from datasets.mit_bih.mit_bih import MITBIHDataset

from .ecg_transformer import ECGTransformer


class CLEP(ECGTransformer):
    def __init__(
        self,
        *args,
        threshold=0.5,
        waves="PRT",
        normalize_loss: bool = True,
        symbol_embedding_dim=1536,
        symbol_embedding_path=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.threshold = threshold
        self.waves = waves
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

        # b, c, w, d: batch_size, lead_num, cls_token_num, embedding_dim
        x = self.fc(
            # b * c, w, d: batch_size * lead_num, cls_token_num, embedding_dim
            self.transformer_forward(x, data["attention_mask"])[:, : self.cls_token_num]
        ).reshape(*x.shape[:2], self.cls_token_num, -1)

        pred = []
        for batch_ind, leads in enumerate(data["signal_name"]):
            cur_pred = []
            for lead_ind, lead in enumerate(leads):
                cur_p = []
                for symbol in MITBIHDataset.SymbolSuperClasses:
                    symbol_embedding = []
                    for i, wave in enumerate(self.waves):
                        if wave in self.symbol_embedding[lead][symbol]:
                            # d: embedding_dim
                            symbol_embedding.append(
                                self.symbol_embedding[lead][symbol][wave]
                            )

                    if self.wave_num_cls_token:
                        cur_x = []
                        for i, wave in enumerate(self.waves):
                            if wave in self.symbol_embedding[lead][symbol]:
                                # d: embedding_dim
                                cur_x.append(x[batch_ind, lead_ind, i, :])

                        # w * d: wave_kind_num * embedding_dim
                        cur_x = torch.cat(cur_x, dim=-1)

                        # w * d: wave_kind_num * embedding_dim
                        symbol_embedding = torch.cat(symbol_embedding, dim=-1)
                    else:
                        # d: embedding_dim
                        cur_x = x[batch_ind, lead_ind, 0, :]
                        # d: embedding_dim
                        symbol_embedding = torch.stack(symbol_embedding).mean(0)

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

        target = data["target"][..., None]
        if self.multi_label:
            target = target.expand(-1, -1, x.shape[1])
            pred = pred.sigmoid()
            loss = self.loss(pred, target.to(pred.dtype))
            # super_class_pred = torch.zeros_like(target, dtype=pred.dtype)
            # for batch_ind in range(target.shape[0]):
            #     for lead_ind in range(target.shape[2]):
            #         for symbol_ind in range(target.shape[1]):
            #             if pred[batch_ind, symbol_ind, lead_ind] > self.threshold:
            #                 symbol = MITBIHDataset.SymbolClasses[symbol_ind]
            #                 super_symbol_ind = (
            #                     MITBIHDataset.SymbolClassToSuperClassIndex[symbol]
            #                 )
            #                 super_class_pred[
            #                     batch_ind, super_symbol_ind, lead_ind
            #                 ] = pred[batch_ind, symbol_ind, lead_ind]
        else:
            target = target.expand(-1, x.shape[1])
            loss = self.loss(pred, target)
            # pred, pred_ind = pred.max(dim=1)
            # super_class_pred = pred.new_zeros(
            #     target.shape[0], MITBIHDataset.SymbolSuperClassesNum, target.shape[1]
            # )
            # for batch_ind in range(target.shape[0]):
            #     for lead_ind in range(target.shape[2]):
            #         symbol = MITBIHDataset.SymbolClasses[pred_ind[batch_ind, lead_ind]]
            #         super_symbol_ind = MITBIHDataset.SymbolClassToSuperClassIndex[
            #             symbol
            #         ]
            #         super_class_pred[batch_ind, super_symbol_ind, lead_ind] = pred[
            #             batch_ind, lead_ind
            #         ]

        return {"log_dict": {"loss": loss}, "pred": pred, "target": target}
