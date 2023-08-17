import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .ecg_conv_transformer import ECGConvTransformer, ECGConvTransformerWithChannel


class ECGConvTransformerLanguageModel(ECGConvTransformer):
    def __init__(
        self,
        description_dim=2048,
        description_model="Deci/DeciCoder-1b",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.description_dim = description_dim
        self.description_embedding = torch.nn.Linear(
            self.embedding_dim_factor * self.embedding_dim, description_dim
        )
        self.description_model = AutoModelForCausalLM.from_pretrained(description_model)
        self.description_tokenizer = AutoTokenizer.from_pretrained(description_model)
        for p in self.description_model.parameters():
            p.requires_grad = False

    def extract_feat(self, x, attention_mask):
        x = self.description_embedding(
            self.encoder_forward(x, attention_mask)[:, : self.cls_token_num]
        ).reshape(*x.shape[:2], self.cls_token_num, -1)
        return x

    def description_forward(self, x, data):
        description = torch.cat(
            [
                x,
                self.description_model.base_model.embed_tokens(data["description"])[
                    :, None
                ].expand(-1, x.shape[1], -1, -1),
            ],
            dim=-2,
        )
        description_attention_mask = data["description_attention_mask"][:, None].expand(
            -1, x.shape[1], -1
        )
        description_attention_mask = torch.cat(
            [
                description_attention_mask.new_ones((*x.shape[:2], 1)),
                description_attention_mask,
            ],
            dim=-1,
        )
        description_labels = data["description"][:, None].expand(-1, x.shape[1], -1)
        description_labels = torch.cat(
            [
                description_labels.new_full(
                    (*description_labels.shape[:2], x.shape[2]), -100
                ),
                description_labels,
            ],
            dim=-1,
        )
        outputs = self.description_model(
            inputs_embeds=description.reshape((-1, *description.shape[2:])),
            attention_mask=description_attention_mask.reshape(
                (-1, *description_attention_mask.shape[2:])
            ),
            labels=description_labels.reshape((-1, *description_labels.shape[2:])),
        )
        return outputs

    def calculate_pred(self, outputs, x):
        pred = outputs.logits[:, x.shape[2] :].argmax(dim=-1)
        pred = [self.description_tokenizer.decode(p) for p in pred]
        return pred

    def calculate_target(self, description, x):
        target = []
        for i in range(x.shape[0]):
            target.extend(
                [self.description_tokenizer.decode(description[i])] * x.shape[1]
            )
        return target

    def forward(self, data):
        x = self.calculate_x(data)
        x, attention_mask = self.embedding(x=x, **data)
        x = self.extract_feat(x, attention_mask)

        outputs = self.description_forward(x, data)

        pred = self.calculate_pred(outputs, x)
        target = self.calculate_target(data["description"], x)

        return {"log_dict": {"loss": outputs.loss}, "pred": pred, "target": target}


class ECGConvTransformerWithChannelLanguageModel(
    ECGConvTransformerLanguageModel, ECGConvTransformerWithChannel
):
    pass
