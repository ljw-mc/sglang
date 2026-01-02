# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Copyright 2024 Cohere and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/models/commandr.py#L1

# This file is based on the LLama model definition file in transformers
"""PyTorch Cohere model."""
from transformers import Cohere2ForCausalLM, CohereForCausalLM # TODO remove later, here for reference

from typing import Iterable, Optional, Tuple

import torch
import torch.utils.checkpoint # probably can remove
from torch import nn
from torch.nn.parameter import Parameter # tags a Tensor as a parameter, allows model to load in a weight/parameter
from transformers import PretrainedConfig # contains all the hyperparameters of the model

from sglang.srt.distributed import (
    get_tensor_model_parallel_rank, # GPU 0 returns 0, GPU 1 return 1 ...
    get_tensor_model_parallel_world_size, # How many GPUs running in parallel --tp 4 means there are 4 GPUs
)
from sglang.srt.layers.activation import SiluAndMul # SiluAndMul = SwiGLU activation in Cohere

from sglang.srt.layers.linear import (
    MergedColumnParallelLinear, # ColumnParallelLinear but they use the same input X, but several different weights, A, B, C
                                # previously, compute X @ A, X @ B, X @ C, 3x different kernel launches
                                # merge A, B, C together into ABC
                                # then only do one matmul y = X @ ABC
                                # and then y[0:aaa] = X @ A
                                # and then y[aaa:bbb] = X @ B
                                # do everything in one matmul
    QKVParallelLinear, # this is mostly used for GroupedQueryAttention / MultiQueryAttention where you have fewer KV heads than Qs
    RowParallelLinear, # splits weights by row. used right after ColumnParallelLinear to save on communication
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from sglang.srt.utils import add_prefix, get_compiler_backend, set_weight_attrs


@torch.compile(backend=get_compiler_backend())
def layer_norm_func(hidden_states, weight, variance_epsilon):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    mean = hidden_states.mean(-1, keepdim=True)
    variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
    hidden_states = (hidden_states - mean) * torch.rsqrt(variance + variance_epsilon)
    hidden_states = weight.to(torch.float32) * hidden_states
    return hidden_states.to(input_dtype)


class LayerNorm(nn.Module):
    def __init__(self, param_shape: Tuple | int, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(param_shape)) # turns out the CohereModel actually does have a weight parameter
        self.variance_epsilon = eps
        set_weight_attrs(self.weight, {"weight_loader": self.weight_loader})

    # TODO: Does the forward signature need a residuals? It is not in the CohereLayerNorm impl on HF.
    # It seems like residuals is passed in and passed out without being used and is also not in CohereLayerNorm
    def forward(self, hidden_states, residuals=None):
        hidden_states = layer_norm_func(
            hidden_states, self.weight, self.variance_epsilon
        )
        return hidden_states, residuals

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        tp_rank = get_tensor_model_parallel_rank()
        shard_dim = 0 if param.dim() != 1 else None
        param_data = param.data
        if shard_dim is not None:
            shard_size = param_data.shape[shard_dim]
            start_idx = tp_rank * shard_size
            loaded_weight = loaded_weight.narrow(shard_dim, start_idx, shard_size)
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


# Copied from transformers.models.llama.modeling_llama.LlamaMLP Llama->Cohere
class CohereMLP(nn.Module):
    def __init__(
        self,
        config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_up_proj = MergedColumnParallelLinear(
            self.hidden_size,
            [self.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x

# TODO: need to support sliding window attention here.
# TODO: seems like Cohere2Attention does not have QK normalization
#       whereas CohereAttention has self.use_qk_norm = config.use_qk_norm and conditionally
#       and conditionally creates self.q_norm, self.k_norm
# TODO: Cohere2Attention applies RoPE only when there is sliding windows
#       CohereAttention uses RoPE after QK norm if used.
#       Cohere2Attention does .view(hidden_shape).transpose(1, 2) immediate after projection
#       CohereAttention does .view first, then QK norm, then transpose
class CohereAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_tensor_model_parallel_world_size()
        self.config = config
        self.attention_dropout = config.attention_dropout # does this get used?
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.num_heads = self.total_num_heads // tp_size
        self.head_dim = self.hidden_size // self.total_num_heads
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = getattr(
            config, "model_max_length", None
        ) or getattr(config, "max_position_embeddings", 8192)
        self.rope_theta = config.rope_theta
        self.rope_scaling = getattr(config, "rope_scaling", None)
        self.use_qk_norm = getattr(config, "use_qk_norm", False)
        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            base=self.rope_theta,
            rope_scaling=self.rope_scaling,
            is_neox_style=False,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )
        if self.use_qk_norm:
            self.q_norm = LayerNorm(
                param_shape=(self.num_heads, self.head_dim), eps=config.layer_norm_eps
            )
            self.k_norm = LayerNorm(
                param_shape=(self.num_kv_heads, self.head_dim),
                eps=config.layer_norm_eps,
            )

    def _apply_qk_norm(self, q, k):
        q = q.view(*q.shape[:-1], -1, self.head_dim)
        k = k.view(*k.shape[:-1], -1, self.head_dim)
        q, _ = self.q_norm(q)
        k, _ = self.k_norm(k)
        q = q.view(*q.shape[:-2], -1)
        k = k.view(*k.shape[:-2], -1)
        return q, k

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.use_qk_norm:
            q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class CohereDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = CohereAttention( # cohere attention
            config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )

        self.mlp = CohereMLP( # a cohere linear layer blocks
            config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = LayerNorm( # cohere layernorm blocks
            param_shape=(config.hidden_size), eps=config.layer_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        residual = hidden_states
        hidden_states, residual = self.input_layernorm(hidden_states, residual) # pre-attn layernorm (I see)
        hidden_states_attention = self.self_attn( # and then self attention
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states_mlp = self.mlp(hidden_states) # mlp
        # Add everything together
        hidden_states = residual + hidden_states_attention + hidden_states_mlp # residuals

        return hidden_states, residual


class CohereModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size # does self.vocab_size ever get used?
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )
        # instantiates like `config.num_hidden_layers` many Cohere decoder blocks
        self.layers = nn.ModuleList(
            [
                CohereDecoderLayer(
                    config,
                    i, # layer id
                    quant_config=quant_config,
                    prefix=add_prefix(f"layers.{i}", prefix),
                )
                for i in range(config.num_hidden_layers)
            ]
        )
        # they have a layernorm layer
        self.norm = LayerNorm(
            param_shape=(config.hidden_size), eps=config.layer_norm_eps
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids) # convert the token IDs to actual embeddings
        residual = None # no residuals
        for i in range(len(self.layers)): # for each decoder block, 
            layer = self.layers[i] # grab layer
            hidden_states, residual = layer( # pass variable through layer
                positions, # positions
                hidden_states, # hidden states after passing through i-1 decoder blocks
                forward_batch, # whatever this is
                residual,
            )
        hidden_states, _ = self.norm(hidden_states, residual) # normalizes at the end
        return hidden_states # return final hidden states, to be passed onto the logits processor


class CohereForCausalLM(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig, # from HF transformers
        quant_config: Optional[QuantizationConfig] = None, # you can choose your quantization methods
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config # set the model's config from HF
        self.quant_config = quant_config # we have the quantization configs
        self.logits_processor = LogitsProcessor(config) # we also have a logits processor from the HF config
        self.model = CohereModel(
            config, quant_config, prefix=add_prefix("model", prefix)
        ) # instantiate the model

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
        )
        return self.logits_processor(
            input_ids, hidden_states, self.model.embed_tokens, forward_batch
        )
    # TODO figure out what this does later
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params = set()
        for name, loaded_weight in weights:
            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # lm_head is not used in vllm as it is tied with embed_token.
                # To prevent errors, skip loading lm_head.weight.
                if "lm_head.weight" in name:
                    continue
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)


class Cohere2ForCausalLM(CohereForCausalLM):
    # we need to instantiate CohereModel
    # in the CohereModel, we might need to have it where we might need to add in sliding windows parameters from HF config
    # this will let us instantiate RadixAttention differently depending on which layer we are at.
    # the question is how similar is the CohereModel for Cohere2 vs Cohere1 and whether we need an entirely new class for Cohere2 
    # probably will take 2.5 weeks to finish
    pass
        
        


EntryClass = [CohereForCausalLM, Cohere2ForCausalLM]
