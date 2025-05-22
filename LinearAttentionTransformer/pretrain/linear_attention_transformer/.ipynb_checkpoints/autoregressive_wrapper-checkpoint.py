from functools import partial
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from linear_attention_transformer.autopadder import Autopadder

def top_p(logits, thres = 0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cum_probs > (1 - thres)
    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
    sorted_indices_to_remove[:, 0] = 0

    sorted_logits[sorted_indices_to_remove] = float('-inf')
    return sorted_logits.scatter(1, sorted_indices, sorted_logits)

def top_k(logits, thres = 0.9):
    k = int((1 - thres) * logits.shape[-1])
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

class AutoregressiveWrapper(nn.Module):
    def __init__(self, net, ignore_index = -100, pad_value = 0):
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = Autopadder(net)
        self.max_seq_len = net.max_seq_len

    @torch.no_grad()
    def generate(self, start_tokens, seq_len, eos_token=None, temperature=1., filter_logits_fn=top_k, filter_thres=0.9, **kwargs):
        was_training = self.net.training
        num_dims = len(start_tokens.shape)

        if num_dims == 2:  # 2次元の場合はバッチ次元を追加
            start_tokens = start_tokens.unsqueeze(0)

        b, feature_dim, t = start_tokens.shape

        self.net.eval()
        out = start_tokens
        input_mask = kwargs.pop('input_mask', None)

        if input_mask is None:
            input_mask = torch.full((b, feature_dim, t), True, dtype=torch.bool, device=start_tokens.device)

        for _ in range(seq_len):
            x = out[:, :, -self.max_seq_len:]  # 最大シーケンス長を切り取る
            input_mask = input_mask[:, :, -self.max_seq_len:]

            logits = self.net(x, input_mask=input_mask, **kwargs)[:, :, -1, :]  # 各特徴次元の次のトークンを予測
            filtered_logits = filter_logits_fn(logits, thres=filter_thres)
            probs = F.softmax(filtered_logits / temperature, dim=-1)
            sample = torch.multinomial(probs, 1)

            out = torch.cat((out, sample.unsqueeze(-1)), dim=-1)
            input_mask = F.pad(input_mask, (0, 1), value=True)

            if eos_token is not None and (sample == eos_token).all():
                break

        out = out[:, :, t:]  # 初期トークンを削除

        if num_dims == 2:  # 元の次元数に戻す
            out = out.squeeze(0)

        self.net.train(was_training)
        return out

    def forward(self, x, return_loss=False, **kwargs):
        pad = partial(pad_sequence, batch_first=True, padding_value=self.pad_value)

        if not return_loss:
            if not isinstance(x, torch.Tensor):
                x = pad(x)
            return self.net(x, **kwargs)

        if isinstance(x, torch.Tensor):
            xi = x[:, :, :-1]  # 入力（最後のタイムステップを除外）
            xo = x[:, :, 1:]   # ターゲット（最初のタイムステップを除外）

            # マスクの処理（次元を調整）
            mask = kwargs.pop('input_mask', None)
            if mask is not None and mask.shape[2] == x.shape[2]:
                mask = mask[:, :, :-1]
                kwargs.update(input_mask=mask)
        else:
            xi = pad([t[:, :-1] for t in x])  # 入力シーケンス
            xo = pad([t[:, 1:] for t in x])  # ターゲットシーケンス

        out = self.net(xi, **kwargs)

        # クロスエントロピー損失計算（3次元データ対応）
        loss = F.cross_entropy(out.transpose(2, 3), xo, ignore_index=self.ignore_index)
        return loss
