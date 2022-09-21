import logging
import math
import os
from typing import Mapping, Optional

import datasets
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from transformers import AutoModel, TrainingArguments, Trainer, \
    EarlyStoppingCallback, AutoTokenizer, DataCollatorWithPadding

from kaggle_ell.solution import Solution
from kaggle_ell.solution_factory import SolutionFactory
from kaggle_ell.solutions.text2text import get_train_test_split
from kaggle_ell.solutions.transformer import Transformer
from kaggle_ell.solutions.transformer_finetune import log_result

logger = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x

def angle_defn(pos, i, d_model_size):
    angle_rates = 1 / torch.pow(10000, (2 * (i // 2)) / d_model_size)
    return pos * angle_rates

def positional_encoding(position, d_model_size, dtype):
    # create the sinusoidal pattern for the positional encoding
    angle_rads = angle_defn(
        torch.arange(position, dtype=dtype).unsqueeze(1),
        torch.arange(d_model_size, dtype=dtype).unsqueeze(0),
        d_model_size,
    )

    sines = torch.sin(angle_rads[:, 0::2])
    cosines = torch.cos(angle_rads[:, 1::2])

    pos_encoding = torch.cat([sines, cosines], dim=-1)
    return pos_encoding

class Qeruy2Label(nn.Module):
    def __init__(self, model_cfg):
        """[summary]

        Args:
            backbone ([type]): backbone model.
            transfomer ([type]): transformer model.
            num_class ([type]): number of classes. (80 for MSCOCO).
        """
        super().__init__()
        self.num_class = 6
        self.backbone = AutoModel.from_pretrained(model_cfg.backbone)

        hidden_dim = self.backbone.config.hidden_size

        self.transformer = Transformer(d_model=hidden_dim,
                                       num_encoder_layers=1,
                                       num_decoder_layers=1,
                                       normalize_before=False,
                                       nhead=1,
                                       rm_self_attn_dec=False,
                                       rm_first_self_attn=False)


        # assert not (self.ada_fc and self.emb_fc), "ada_fc and emb_fc cannot be True at the same time."


        self.input_proj = nn.Identity() # nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.query_embed = nn.Embedding(self.num_class, hidden_dim)
        self.fc = GroupWiseLinear(self.num_class, hidden_dim, bias=True)

        # TODO: revisit max_position_embeddings here
        max_pos_emb = self.backbone.config.max_position_embeddings
        self.pos_encoding = positional_encoding(max_pos_emb, hidden_dim, torch.float)

        self.loss_fn = nn.MSELoss()

    def forward(self,
                    input_ids: Optional[torch.Tensor] = None,
                    attention_mask: Optional[torch.Tensor] = None,
                    token_type_ids: Optional[torch.Tensor] = None,
                    labels: Optional[torch.Tensor] = None):

        device = input_ids.device

        position_ids = torch.arange(0, input_ids.size()[-1], dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_ids.size()[-1])
        pos_embeds = self.pos_encoding[position_ids, :].to(device)

        backbone_out = self.backbone(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     token_type_ids=token_type_ids,
                                     ) # TODO: currently not using position embedding. Not clear if needed.

        query_input = self.query_embed.weight
        hs = self.transformer(self.input_proj(backbone_out.last_hidden_state),
                              query_input, pos_embeds, mask=attention_mask)[0]  # B,K,d
        out = self.fc(hs)
        # import ipdb; ipdb.set_trace()

        if labels is None:
            loss = None
        else:
            loss = self.loss_fn(out, labels)

        return ((loss,) + (out, )) if loss is not None else out

    def finetune_params(self):
        from itertools import chain
        return chain(self.transformer.parameters(), self.fc.parameters(), self.input_proj.parameters(),
                     self.query_embed.parameters())

    def load_backbone(self, path):
        print("=> loading checkpoint '{}'".format(path))


def convert_to_features(example, tokenizer, data_cfg, target_cols):
    features = tokenizer(example['full_text'],
                              padding=False,
                              max_length=data_cfg.source_max_length,
                              truncation=True)

    if 'cohesion' in example:
        label = [example[x] for x in target_cols]

        return {
            **features,
            'label': label
        }
    else:
        return features


def preprocess_train(train_cfg, train_df, target_cols):
    Fold = MultilabelStratifiedKFold(n_splits=train_cfg.n_fold, shuffle=True, random_state=train_cfg.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(train_df, train_df[target_cols])):
        train_df.loc[val_index, 'fold'] = int(n)
    train_df['fold'] = train_df['fold'].astype(int)
    return datasets.Dataset.from_pandas(train_df.reset_index(drop=True))

def transform_datasets_for_train(split_ds, tokenizer, data_cfg, target_cols):
    columns = ['input_ids', 'token_type_ids', 'attention_mask', 'label' ]
    encoded_ds = {}
    for key, ds in split_ds.items():
        #remove_cols = set(ds.features.keys()) - set(['input_ids', 'attention_mask', 'label', 'token_type_ids'])
        #remove_columns = remove_cols
        encoded_ds[key] = ds.map(lambda x: convert_to_features(x, tokenizer, data_cfg, target_cols))
        encoded_ds[key].set_format(type='torch', columns=columns)

    return encoded_ds

def train_loop(model, train_ds, fold_idx, train_cfg, data_cfg, artifacts_path, device, tokenizer, target_cols):
    raw_ds_dict = get_train_test_split(train_ds, fold_idx, data_cfg)
    ds_dict = transform_datasets_for_train(raw_ds_dict, tokenizer, data_cfg, target_cols)

    train_args = TrainingArguments(
        f"model{fold_idx}",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        learning_rate=train_cfg.learning_rate,
        per_device_train_batch_size=train_cfg.train_batch_size,
        per_device_eval_batch_size=train_cfg.eval_batch_size,
        num_train_epochs=train_cfg.epochs,
        weight_decay=train_cfg.weight_decay,
        warmup_ratio=train_cfg.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=1,
        log_level='error',
        optim=train_cfg.optim,
        fp16=train_cfg.fp16,
        gradient_checkpointing=train_cfg.gradient_checkpointing,
        gradient_accumulation_steps=train_cfg.gradient_accumulation_steps,
        lr_scheduler_type=train_cfg.lr_scheduler_type
    )

    trainer = Trainer(
        model,
        train_args,
        train_dataset=ds_dict["train"],
        eval_dataset=ds_dict["test"],
        tokenizer=tokenizer,
        # compute_metrics=lambda x: compute_metrics(tokenizer, x),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=train_cfg.early_stopping_patience)],
        data_collator=DataCollatorWithPadding(tokenizer)
    )

    trainer.evaluate()
    trainer.train()

    raw_preds = np.array(trainer.predict(ds_dict["test"]).predictions)
    oof_df = raw_ds_dict["test"].to_pandas()
    preds_df = pd.DataFrame({f'pred_{k}': raw_preds[:, idx] for idx, k in enumerate(target_cols)}, index=oof_df.index)
    oof_df = pd.concat([oof_df, preds_df], axis=1)
    #oof_df['pred'] = oof_preds
    return oof_df

@SolutionFactory.register('query2label')
class Query2Label(Solution):
    def do_train(self, train_data: pd.DataFrame, data_cfg: Mapping, train_cfg: Mapping, model_cfg: Mapping,
                 env_cfg: Mapping):

        tokenizer = AutoTokenizer.from_pretrained(model_cfg.backbone)

        raw_train_df = self.competition_data_manager.load_train_data()

        target_cols = self.competition_data_manager.LABEL_COLUMNS

        train_ds = preprocess_train(train_cfg, raw_train_df, target_cols)

        tokenizer.save_pretrained(os.path.join(env_cfg.artifacts_path, 'tokenizer'))

        oof_df = pd.DataFrame()
        for fold in range(train_cfg.n_fold):
            if fold in train_cfg.trn_fold:
                model = Qeruy2Label(model_cfg)

                _oof_df = train_loop(model, train_ds, fold, train_cfg, data_cfg, env_cfg.artifacts_path, env_cfg.device,
                                     tokenizer, target_cols)
                oof_df = pd.concat([oof_df, _oof_df])
                logger.info(f"========== fold: {fold} result ==========")
                log_result(_oof_df, target_cols)
        oof_df = oof_df.reset_index(drop=True)
        logger.info(f"========== CV ==========")
        log_result(oof_df, target_cols)
        oof_df.to_pickle(os.path.join(env_cfg.artifacts_path, 'oof_df.pkl'))

        pass

    def do_predict(self, input_data: pd.DataFrame, data_cfg: Mapping, inference_cfg: Mapping, model_cfg: Mapping,
                   env_cfg: Mapping):
        pass