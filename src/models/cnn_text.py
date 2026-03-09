# src/models/cnn_text.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class ResNetBlock1D(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1, kernel_size=3):
        super().__init__()
        if not subsample:
            c_out = c_in

        pad = kernel_size // 2

        self.net = nn.Sequential(
            nn.Conv1d(
                c_in,
                c_out,
                kernel_size=kernel_size,
                padding=pad,
                stride=1 if not subsample else 2,
                bias=False,
            ),
            nn.BatchNorm1d(c_out),
            act_fn(),
            nn.Conv1d(
                c_out,
                c_out,
                kernel_size=kernel_size,
                padding=pad,
                bias=False,
            ),
            nn.BatchNorm1d(c_out),
        )

        self.downsample = nn.Conv1d(c_in, c_out, kernel_size=1, stride=2) if subsample else None
        self.act_fn = act_fn()

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        out = self.act_fn(out)
        return out


class PreActResNetBlock1D(nn.Module):
    def __init__(self, c_in, act_fn, subsample=False, c_out=-1, kernel_size=3):
        super().__init__()
        if not subsample:
            c_out = c_in

        pad = kernel_size // 2

        self.net = nn.Sequential(
            nn.BatchNorm1d(c_in),
            act_fn(),
            nn.Conv1d(
                c_in,
                c_out,
                kernel_size=kernel_size,
                padding=pad,
                stride=1 if not subsample else 2,
                bias=False,
            ),
            nn.BatchNorm1d(c_out),
            act_fn(),
            nn.Conv1d(
                c_out,
                c_out,
                kernel_size=kernel_size,
                padding=pad,
                bias=False,
            ),
        )

        self.downsample = nn.Sequential(
            nn.BatchNorm1d(c_in),
            act_fn(),
            nn.Conv1d(c_in, c_out, kernel_size=1, stride=2, bias=False),
        ) if subsample else None

    def forward(self, x):
        z = self.net(x)
        if self.downsample is not None:
            x = self.downsample(x)
        out = z + x
        return out


resnet_blocks_by_name_1d = {
    "ResNetBlock": ResNetBlock1D,
    "PreActResNetBlock": PreActResNetBlock1D,
}

act_fn_by_name = {
    "relu": nn.ReLU,
    "elu": nn.ELU,
    "gelu": nn.GELU,
}


class TextResNetCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_classes=3,
        pad_idx=0,
        emb_dim=128,
        num_blocks=(2, 2, 2),
        c_hidden=(64, 128, 256),
        act_fn_name="relu",
        block_name="ResNetBlock",
        dropout=0.1,
        kernel_size=3,
        pool="max",
    ):
        super().__init__()

        if act_fn_name not in act_fn_by_name:
            raise ValueError(f"Unknown act_fn_name={act_fn_name}. Choose from {list(act_fn_by_name.keys())}")
        if block_name not in resnet_blocks_by_name_1d:
            raise ValueError(f"Unknown block_name={block_name}. Choose from {list(resnet_blocks_by_name_1d.keys())}")
        if len(num_blocks) != len(c_hidden):
            raise ValueError("num_blocks and c_hidden must have the same length")

        self.hparams = {
            "num_classes": num_classes,
            "c_hidden": list(c_hidden),
            "num_blocks": list(num_blocks),
            "act_fn_name": act_fn_name,
            "act_fn": act_fn_by_name[act_fn_name],
            "block_class": resnet_blocks_by_name_1d[block_name],
            "kernel_size": kernel_size,
            "pool": pool,
            "dropout": dropout,
        }

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)

        self._create_network(emb_dim=emb_dim)
        self._init_params()

    def _create_network(self, emb_dim):
        c_hidden = self.hparams["c_hidden"]
        act_fn = self.hparams["act_fn"]
        block_class = self.hparams["block_class"]
        kernel_size = self.hparams["kernel_size"]

        if block_class == PreActResNetBlock1D:
            self.input_net = nn.Sequential(
                nn.Conv1d(emb_dim, c_hidden[0], kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
            )
        else:
            self.input_net = nn.Sequential(
                nn.Conv1d(emb_dim, c_hidden[0], kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
                nn.BatchNorm1d(c_hidden[0]),
                act_fn(),
            )

        blocks = []
        for block_idx, block_count in enumerate(self.hparams["num_blocks"]):
            for bc in range(block_count):
                subsample = (bc == 0 and block_idx > 0)
                c_in = c_hidden[block_idx if not subsample else (block_idx - 1)]
                blocks.append(
                    block_class(
                        c_in=c_in,
                        act_fn=act_fn,
                        subsample=subsample,
                        c_out=c_hidden[block_idx],
                        kernel_size=kernel_size,
                    )
                )
        self.blocks = nn.Sequential(*blocks)

        if self.hparams["pool"] == "max":
            self.pool = nn.AdaptiveMaxPool1d(1)
        elif self.hparams["pool"] == "avg":
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError("pool must be 'max' or 'avg'")

        self.output_net = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(self.hparams["dropout"]),
            nn.Linear(c_hidden[-1], self.hparams["num_classes"]),
        )

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity=self.hparams["act_fn_name"])
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x_ids):
        x = self.embedding(x_ids)
        x = x.transpose(1, 2)
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.pool(x)
        x = self.output_net(x)
        return x

    @property
    def device(self):
        return next(self.parameters()).device
