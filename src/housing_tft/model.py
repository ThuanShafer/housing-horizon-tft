"""A compact Temporal Fusion Transformer.

The blocks follow Lim, Arık, Loeff and Pfister (2021): a gated residual
network, variable selection, an LSTM encoder–decoder, static enrichment,
interpretable multi-head attention, and a position-wise gated residual.
The paper's quantile head is a five-class head here, because the requested
output is one of the five stability tiers at each horizon.

Dropout is inside the residual blocks and is off at evaluation. The LSTM
itself is ``nn.LSTM``. ``manual_lstm`` matches that cell so the portal can
recompute a forecast without PyTorch.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from housing_tft.spec import (
    FUTURE_NAMES,
    HIST_NAMES,
    KNOWN_CAT,
    KNOWN_NUM,
    OBS_CAT,
    OBS_NUM,
    STATIC_CAT,
    STATIC_NAMES,
    STATIC_NUM,
    VOCABS,
)


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mean = x.mean(dim=-1, keepdim=True)
    var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
    return (x - mean) * torch.rsqrt(var + eps) * weight + bias


def manual_lstm(x, h, c, weight_ih, weight_hh, bias_ih, bias_hh):
    """Unrolled LSTM, gate order input, forget, cell, output. Matches ``nn.LSTM``."""
    outputs = []
    for t in range(x.shape[1]):
        gates = x[:, t] @ weight_ih.T + bias_ih + h @ weight_hh.T + bias_hh
        i, f, g, o = gates.chunk(4, dim=-1)
        i, f, g, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(g), torch.sigmoid(o)
        c = f * c + i * g
        h = o * torch.tanh(c)
        outputs.append(h)
    return torch.stack(outputs, dim=1), h, c


class GRN(nn.Module):
    def __init__(self, d_in: int, d_out: int, d_context: int = 0, d_hidden: int = 48, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.fc_c = nn.Linear(d_context, d_hidden, bias=False) if d_context else None
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.glu = nn.Linear(d_out, 2 * d_out)
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else None
        self.norm_w = nn.Parameter(torch.ones(d_out))
        self.norm_b = nn.Parameter(torch.zeros(d_out))
        self.dropout = dropout
        self.eps = 1e-5

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        residual = x if self.skip is None else self.skip(x)
        hidden = self.fc1(x)
        if self.fc_c is not None:
            if context is None:
                raise ValueError("this GRN expects a context vector")
            if context.shape[:-1] != x.shape[:-1]:
                while context.dim() < x.dim():
                    context = context.unsqueeze(-2)
                context = context.expand(*x.shape[:-1], context.shape[-1])
            hidden = hidden + self.fc_c(context)
        hidden = torch.nn.functional.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = torch.nn.functional.dropout(hidden, self.dropout, self.training)
        a, b = self.glu(hidden).chunk(2, dim=-1)
        return layer_norm(residual + a * torch.sigmoid(b), self.norm_w, self.norm_b, self.eps)


class BatchedGRN(nn.Module):
    """One GRN per variable, applied on [..., n, d_in]."""

    def __init__(self, n: int, d_in: int, d_out: int, d_hidden: int, dropout: float):
        super().__init__()
        self.w1 = nn.Parameter(torch.empty(n, d_in, d_hidden))
        self.b1 = nn.Parameter(torch.zeros(n, d_hidden))
        self.w2 = nn.Parameter(torch.empty(n, d_hidden, d_out))
        self.b2 = nn.Parameter(torch.zeros(n, d_out))
        self.wg = nn.Parameter(torch.empty(n, d_out, 2 * d_out))
        self.bg = nn.Parameter(torch.zeros(n, 2 * d_out))
        self.same = d_in == d_out
        if not self.same:
            self.skip_w = nn.Parameter(torch.empty(n, d_in, d_out))
            self.skip_b = nn.Parameter(torch.zeros(n, d_out))
        self.ln_g = nn.Parameter(torch.ones(n, d_out))
        self.ln_b = nn.Parameter(torch.zeros(n, d_out))
        self.dropout = dropout
        self.eps = 1e-5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for index in range(self.w1.shape[0]):
            nn.init.xavier_uniform_(self.w1[index])
            nn.init.xavier_uniform_(self.w2[index])
            nn.init.xavier_uniform_(self.wg[index])
            if not self.same:
                nn.init.xavier_uniform_(self.skip_w[index])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.einsum("...ni,nio->...no", x, self.w1) + self.b1
        hidden = torch.nn.functional.elu(hidden)
        hidden = torch.einsum("...ni,nio->...no", hidden, self.w2) + self.b2
        hidden = torch.nn.functional.dropout(hidden, self.dropout, self.training)
        gates = torch.einsum("...ni,nio->...no", hidden, self.wg) + self.bg
        a, b = gates.chunk(2, dim=-1)
        mixed = a * torch.sigmoid(b)
        residual = x if self.same else torch.einsum("...ni,nio->...no", x, self.skip_w) + self.skip_b
        return layer_norm(residual + mixed, self.ln_g, self.ln_b, self.eps)


class VariableSelection(nn.Module):
    def __init__(self, n_vars: int, d_model: int, d_context: int, d_hidden: int, dropout: float):
        super().__init__()
        self.selector = GRN(n_vars * d_model, n_vars, d_context, d_hidden, dropout)
        self.var = BatchedGRN(n_vars, d_model, d_model, d_hidden, dropout)

    def forward(self, values: torch.Tensor, context: torch.Tensor | None = None):
        # values: [B, T, N, D] or [B, N, D]
        squeezed = values.dim() == 3
        if squeezed:
            values = values.unsqueeze(1)
        flat = values.flatten(-2)
        weights = torch.softmax(self.selector(flat, context), dim=-1)
        processed = self.var(values)
        selected = (processed * weights.unsqueeze(-1)).sum(dim=-2)
        if squeezed:
            return selected.squeeze(1), weights.squeeze(1)
        return selected, weights


class NumericEmbed(nn.Module):
    def __init__(self, n: int, d_model: int):
        super().__init__()
        bound = (6.0 / (1.0 + d_model)) ** 0.5
        self.w = nn.Parameter(torch.empty(n, d_model).uniform_(-bound, bound))
        self.b = nn.Parameter(torch.zeros(n, d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.unsqueeze(-1) * self.w + self.b


class CategoricalEmbed(nn.Module):
    def __init__(self, cards: list[int], d_model: int):
        super().__init__()
        self.embs = nn.ModuleList(nn.Embedding(card, d_model) for card in cards)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cols = [emb(x[..., i]) for i, emb in enumerate(self.embs)]
        return torch.stack(cols, dim=-2)


class InterpretableMultiHead(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must divide into n_heads")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.Wq = nn.Parameter(torch.empty(n_heads, d_model, self.d_k))
        self.bq = nn.Parameter(torch.zeros(n_heads, self.d_k))
        self.Wk = nn.Parameter(torch.empty(n_heads, d_model, self.d_k))
        self.bk = nn.Parameter(torch.zeros(n_heads, self.d_k))
        self.Wv = nn.Linear(d_model, self.d_k)
        self.Wo = nn.Linear(self.d_k, d_model)
        self.dropout = dropout
        for head in range(n_heads):
            nn.init.xavier_uniform_(self.Wq[head])
            nn.init.xavier_uniform_(self.Wk[head])

    def forward(self, query: torch.Tensor, context: torch.Tensor, mask: torch.Tensor):
        q = torch.einsum("btd,hdk->bhtk", query, self.Wq) + self.bq[None, :, None, :]
        k = torch.einsum("bsd,hdk->bhsk", context, self.Wk) + self.bk[None, :, None, :]
        scores = torch.einsum("bhtk,bhsk->bhts", q, k) / math.sqrt(self.d_k)
        scores = scores.masked_fill(~mask, -1e9)
        attn = torch.softmax(scores, dim=-1).mean(dim=1)
        attn = torch.nn.functional.dropout(attn, self.dropout, self.training)
        mixed = self.Wo(torch.matmul(attn, self.Wv(context)))
        return mixed, attn


class TemporalFusionTransformer(nn.Module):
    def __init__(self, d_model: int = 32, d_hidden: int = 48, n_heads: int = 4, dropout: float = 0.1,
                 encoder_len: int = 12, decoder_len: int = 12):
        super().__init__()
        self.d_model = d_model
        self.encoder_len = encoder_len
        self.decoder_len = decoder_len
        self.hist_names = list(HIST_NAMES)
        self.future_names = list(FUTURE_NAMES)
        self.static_names = list(STATIC_NAMES)
        self.static_num = NumericEmbed(len(STATIC_NUM), d_model) if STATIC_NUM else None
        self.static_cat = CategoricalEmbed([len(VOCABS[f.name]) + 1 for f in STATIC_CAT], d_model)
        self.obs_num = NumericEmbed(len(OBS_NUM), d_model)
        self.obs_cat = CategoricalEmbed([len(VOCABS[f.name]) + 1 for f in OBS_CAT], d_model)
        self.known_num = NumericEmbed(len(KNOWN_NUM), d_model)
        self.known_cat = CategoricalEmbed([len(VOCABS[f.name]) + 1 for f in KNOWN_CAT], d_model)
        n_static = len(STATIC_NUM) + len(STATIC_CAT)
        n_hist = len(OBS_NUM) + len(OBS_CAT) + len(KNOWN_NUM) + len(KNOWN_CAT)
        n_future = len(KNOWN_NUM) + len(KNOWN_CAT)
        self.vsn_static = VariableSelection(n_static, d_model, 0, d_hidden, dropout)
        self.vsn_hist = VariableSelection(n_hist, d_model, d_model, d_hidden, dropout)
        self.vsn_future = VariableSelection(n_future, d_model, d_model, d_hidden, dropout)
        self.ctx_s = GRN(d_model, d_model, 0, d_hidden, dropout)
        self.ctx_e = GRN(d_model, d_model, 0, d_hidden, dropout)
        self.ctx_h = GRN(d_model, d_model, 0, d_hidden, dropout)
        self.ctx_c = GRN(d_model, d_model, 0, d_hidden, dropout)
        self.lstm = nn.LSTM(d_model, d_model, num_layers=1, batch_first=True)
        hidden = d_model
        self.lstm.bias_ih_l0.data[hidden: 2 * hidden].fill_(1.0)
        self.lstm.bias_hh_l0.data[hidden: 2 * hidden].fill_(1.0)
        self.enrich = GRN(d_model, d_model, d_model, d_hidden, dropout)
        self.attn = InterpretableMultiHead(d_model, n_heads, dropout)
        self.gate1 = nn.Linear(d_model, 2 * d_model)
        self.norm1_w = nn.Parameter(torch.ones(d_model))
        self.norm1_b = nn.Parameter(torch.zeros(d_model))
        self.ff = GRN(d_model, d_model, 0, d_hidden, dropout)
        self.gate2 = nn.Linear(d_model, 2 * d_model)
        self.norm2_w = nn.Parameter(torch.ones(d_model))
        self.norm2_b = nn.Parameter(torch.zeros(d_model))
        self.now_head = nn.Linear(d_model, 5)
        self.head = nn.Linear(d_model, 5)
        mask = torch.zeros(decoder_len, encoder_len + decoder_len, dtype=torch.bool)
        for step in range(decoder_len):
            mask[step, : encoder_len + step + 1] = True
        self.register_buffer("attn_mask", mask)

    def _gated_residual(self, update: torch.Tensor, skip: torch.Tensor, gate: nn.Linear, weight, bias) -> torch.Tensor:
        a, b = gate(update).chunk(2, dim=-1)
        return layer_norm(skip + a * torch.sigmoid(b), weight, bias)

    def forward(self, static_cat, static_num, hist_cat, hist_num, known_cat_hist, known_num_hist, known_cat_fut, known_num_fut):
        static_parts = []
        if self.static_num is not None:
            static_parts.append(self.static_num(static_num))
        static_parts.append(self.static_cat(static_cat))
        static = torch.cat(static_parts, dim=-2)
        static_rep, w_static = self.vsn_static(static, None)
        c_s = self.ctx_s(static_rep)
        c_e = self.ctx_e(static_rep)
        h0 = torch.tanh(self.ctx_h(static_rep)).unsqueeze(0)
        c0 = torch.tanh(self.ctx_c(static_rep)).unsqueeze(0)

        hist = torch.cat([
            self.obs_num(hist_num),
            self.obs_cat(hist_cat),
            self.known_num(known_num_hist),
            self.known_cat(known_cat_hist),
        ], dim=-2)
        future = torch.cat([self.known_num(known_num_fut), self.known_cat(known_cat_fut)], dim=-2)
        hist_sel, w_hist = self.vsn_hist(hist, c_s)
        fut_sel, w_fut = self.vsn_future(future, c_s)
        enc_out, (h_n, c_n) = self.lstm(hist_sel, (h0, c0))
        dec_out, _ = self.lstm(fut_sel, (h_n, c_n))
        enc_e = self.enrich(enc_out, c_e)
        dec_e = self.enrich(dec_out, c_e)
        sequence = torch.cat([enc_e, dec_e], dim=1)
        attn_out, attn = self.attn(dec_e, sequence, self.attn_mask)
        skipped = self._gated_residual(attn_out, dec_e, self.gate1, self.norm1_w, self.norm1_b)
        fed = self.ff(skipped)
        fused = self._gated_residual(fed, skipped, self.gate2, self.norm2_w, self.norm2_b)
        return self.now_head(enc_e[:, -1]), self.head(fused), attn, w_static, w_hist, w_fut


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
