import torch
from torch import nn
import torch.nn.functional as F
from trainpilot.config import TrainPilotconfig
import math


# RMSNorm类，是在模块输入的时候对输入token的隐藏向量进行归一化的关键一步，在归一化之后各个特征之间
# 可能出现数值消失的情况，因为无法捕捉到具体的大小，所以这里引入一个可学习变量，将关键的特征进行适当的放大或者缩小
class RMSNorm(nn.Module):
    def __init__(self, dim : int, eps:float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)

# SwiGLU 门控激活：FFN 内部使用的一种激活方式。
# 与普通 FFN 的区别是多了一个门控单元：
#   - gate_proj(x) 经过 SiLU 激活后，作为门控信号；
#   - 与 up_proj(x) 逐元素相乘，动态调节每个中间维度的通过量；
#   - gate_proj、up_proj 的权重都是可学习的，所以门控值随输入变化；
#   - 门控接近 0 的维度被抑制，接近 1 的维度被放行，从而让网络自己
#     决定哪些特征该保留、哪些该弱化，提升 FFN 的表达能力。
class FFN_SWiGLU(nn.Module):
    def __init__(self, config: TrainPilotconfig, intermediate_size:int = None):
        super().__init__()
        intermediate_size = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, intermediate_size, bias = False)
        self.up_proj = nn.Linear(config.hidden_size, intermediate_size, bias = False)
        self.down_proj = nn.Linear(intermediate_size, config.hidden_size, bias = False)
        self.act_fn = F.silu

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


# RoPR位置编码是将位置信息注入到输入中的一种方式，rope是当前最常用也是被证明性能最好的方法，他的核心思想是将位置信息转变为多维度上的向量旋转
# 每一个向量的旋转角度不同来确定绝对位置，同时他们之间的差异可以简单的由相对位置差异来确定，这里使用的计算方法是llama方式的实现
# 而不是rope原文中的实现，主要的影响只是代码的简洁性和存储空间的简洁性的一些改进，RoPE方法将作为输入前的第一个方法调用，之后携带着位置信息的
# 输入将会依次进入层进行训练或者计算，由于RoPE没有学习的参数，只是纯数学的编码方式，所以这里不需要使用类来定义

def precompute_freqs_cis(dim:int, end:int = (32 * 1024), rope_base:int = 1e6, rope_scaling: dict = None):
    # 基础频率计算
    freqs = 1 / (rope_base ** (torch.arange(0,dim,2).float()/dim))
    # 保留yarn延展选项

    # 输入序列的位置编码
    t = torch.arange(end, device = freqs.device)

    freqs = torch.outer(t, freqs).float()

    freqs_cos =  torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim = -1)
    return freqs_cos, freqs_sin

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)
    q_embed = ((q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))).to(q.dtype)
    k_embed = ((k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))).to(k.dtype)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, num_key_value_heads, head_dim = x.shape
    if n_rep == 1: return x
    return (x[:, :, :, None, :].expand(bs, slen, num_key_value_heads, n_rep, head_dim).reshape(bs, slen, num_key_value_heads * n_rep, head_dim))

# GQA ATTN实现
# falsh attn是一种避免显存过载的函数，他不是将大矩阵直接放到显卡上跑，而是把QKV分块，每次只分出一小块放到显卡上跑，用完就丢
#

class Attention(nn.Module):
    def __init__(self, config:TrainPilotconfig):
        super().__init__()

        # 有关GQA的配置
        self.num_kv_heads = config.num_heads if config.num_kv_heads is None else config.num_kv_heads
        self.num_heads = config.num_heads
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = config.head_dim


        # 四个投影层，主要变化的是投影的维度
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_heads * self.head_dim, config.hidden_size, bias=False)

        # Q,K的norm
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps)

        # 设置dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention') and config.flash_attn

        self.is_causal = True

    def forward(self, x, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        B, T, S = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        xq = xq.view(B, T, self.num_heads, self.head_dim)
        xk = xk.view(B, T, self.num_kv_heads, self.head_dim)
        xv = xv.view(B, T, self.num_kv_heads, self.head_dim)

        xq, xk = self.q_norm(xq), self.k_norm(xk)

        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq, xk, xv = (
            xq.transpose(1, 2),
            repeat_kv(xk, self.n_rep).transpose(1, 2),
            repeat_kv(xv, self.n_rep).transpose(1, 2),
        )
        if self.flash and (T > 1) and (not self.is_causal or past_key_value is None) and (attention_mask is None or torch.all(attention_mask == 1)):
            output = F.scaled_dot_product_attention(
                xq,
                xk,
                xv,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=self.is_causal,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)).float() / math.sqrt(self.head_dim)
            if self.is_causal:
                scores[:, :, :, -T:] += torch.full((T,T), float("-inf"), device=scores.device).triu(1)
            if attention_mask is not None:
                scores += (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * -1e9
            output = self.attn_dropout(F.softmax(scores, dim=-1).type_as(xq)) @ xv
        output = output.transpose(1, 2).reshape(B, T, -1)
        output = self.resid_dropout(self.o_proj(output))
        return output, past_kv
