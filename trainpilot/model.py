"""项目自己的 Dense Transformer 骨干与因果语言模型。

使用本项目 TrainPilotconfig 与 layers；语言模型输入 [B,T]，输出 logits [B,T,V]。
下一 token loss 在训练端统一计算，数据端保留未移位 labels。
基于 nn.Module，不导入外部 MiniMind 或 Hugging Face 模型类。
"""
from trainpilot.layers import RMSNorm, FFN_SWiGLU, Attention, precompute_freqs_cis
from trainpilot.config import TrainPilotconfig
import torch
from torch import nn

# 这里实现的是一个attnblock，是一个关键的组成部分，计算方法还是transformer中的模块方法
class TrainPilotBlock(nn.Module):
    def __init__(self, layer_id: int, config: TrainPilotconfig):
        super().__init__()
        self.layer_id = layer_id
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FFN_SWiGLU(config)

    def forward(self, hidden_states, position_embeddings, past_key_value=None, use_cache=False, attention_mask=None):
        # 1. 保存残差
        residual = hidden_states

        # 2. Attention 子层：先 Norm，再 Attention
        hidden_states, present_key_value = self.self_attn(
            self.input_layernorm(hidden_states),
            position_embeddings,
            past_key_value,
            use_cache,
            attention_mask,
        )

        # 3. Attention 残差
        hidden_states = hidden_states + residual

        # 4. FFN 子层：先 Norm，再 FFN，再加残差
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))

        return hidden_states, present_key_value




class TrainPilotModel(nn.Module):
    def __init__(self, config: TrainPilotconfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.num_layers = config.num_layers

        # 1. 词嵌入
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)

        # 2. N 个独立 Block
        self.layers = nn.ModuleList([
            TrainPilotBlock(l, config) for l in range(self.num_layers)
        ])

        # 3. 最终 RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # 4. 预计算 RoPE，注册为 buffer
        freqs_cos, freqs_sin = precompute_freqs_cis(
            dim=config.head_dim,
            end=config.max_seq_len,                       # ← 改
            rope_base=config.rope_theta,
            rope_scaling=getattr(config, "rope_scaling", None),
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        use_cache=False,
    ):
        B, T = input_ids.shape

        # ---- 处理 KV cache ----
        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = (
            past_key_values[0][0].shape[1]
            if past_key_values[0] is not None
            else 0
        )

        # ---- 位置范围检查 ----
        if start_pos + T > self.config.max_seq_len:
            raise ValueError(
                f"Sequence position out of range: "
                f"start_pos({start_pos}) + T({T}) = {start_pos + T} "
                f"> max_seq_len({self.config.max_seq_len})"
            )

        # ---- Embedding + dropout ----
        hidden_states = self.dropout(self.embed_tokens(input_ids))

        # ---- RoPE buffer 兜底重建 ----
        if self.freqs_cos[0, 0] == 0:
            freqs_cos, freqs_sin = precompute_freqs_cis(
                dim=self.config.head_dim,
                end=self.config.max_seq_len,              # ← 改
                rope_base=self.config.rope_theta,
                rope_scaling=getattr(self.config, "rope_scaling", None),
            )
            self.freqs_cos = freqs_cos.to(hidden_states.device)
            self.freqs_sin = freqs_sin.to(hidden_states.device)

        # ---- 按实际序列长度截取 cos/sin ----
        position_embeddings = (
            self.freqs_cos[start_pos: start_pos + T],
            self.freqs_sin[start_pos: start_pos + T],
        )

        # ---- 逐层前向 ----
        presents = []
        for layer, past_key_value in zip(self.layers, past_key_values):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                past_key_value=past_key_value,
                use_cache=use_cache,
                attention_mask=attention_mask,
            )
            presents.append(present)

        # ---- 最终归一化 ----
        hidden_states = self.norm(hidden_states)

        return hidden_states, presents


# 与lm_head组成为一个完整的模型
class TrainPilotForCausalLM(nn.Module):
    def __init__(self, config: TrainPilotconfig):
        super().__init__()
        self.config = config

        # 1. 骨干模型
        self.model = TrainPilotModel(config)

        # 2. LM Head：hidden_size → vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 3. 权重共享（tie_word_embeddings=True 时）
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids, attention_mask=None):
        # 调用骨干，解包得到隐藏表示
        hidden_states, _ = self.model(input_ids, attention_mask=attention_mask)

        # 经过 LM Head，得到 logits
        logits = self.lm_head(hidden_states)

        return logits
