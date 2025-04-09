from typing import Dict, Optional
from omegaconf import DictConfig

import torch
import torch.nn as nn
from cutie.model.group_modules import GConv2d
from cutie.utils.tensor_utils import aggregate
from cutie.model.transformer.positional_encoding import PositionalEncoding
from cutie.model.transformer.transformer_layers import *
from cutie.model.transformer.object_summarizer import *

class ObjectTokenizer(nn.Module):
    def __init__(self, c_in, c_out, num_slots):
        super().__init__()
        self.num_slots = num_slots
        self.proj_q = nn.Linear(c_in, c_out)
        self.slots = nn.Parameter(torch.randn(1, 1, num_slots, c_out))  # (1, 1, q, C)

    def forward(self, obj_feats):  # obj_feats: (B, N, C, H, W)
        B, N, C, H, W = obj_feats.shape

        # 1. Flatten spatial
        x = obj_feats.view(B, N, C, H * W).transpose(2, 3)  # (B, N, HW, C)
        q = self.proj_q(x)  # (B, N, HW, C)

        # 2. Prepare learnable slots (B, N, q, C)
        slots = self.slots.expand(B, N, -1, -1)

        # 3. Compute attention: (B, N, q, HW)
        attn_logits = torch.einsum('bnqc,bnhc->bnqh', slots, q)  # dot-product attention
        attn = attn_logits.softmax(dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)  # prevent NaNs from full masking

        # 4. Weighted sum: (B, N, q, C)
        token_out = torch.einsum('bnqh,bnhc->bnqc', attn, q)

        return token_out  # (B, N, q, C)


class QueryTransformerBlock(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()

        this_cfg = model_cfg.object_transformer
        self.embed_dim = this_cfg.embed_dim
        self.num_heads = this_cfg.num_heads
        self.num_queries = this_cfg.num_queries
        self.ff_dim = this_cfg.ff_dim


        self.out_from_mask = CrossAttention(self.embed_dim,
                                            self.num_heads,
                                            add_pe_to_qkv=this_cfg.read_from_pixel.add_pe_to_qkv)
        self.ffn = FFN(self.embed_dim, self.ff_dim)

        self.out_from_object = CrossAttention(self.embed_dim,
                                            self.num_heads,
                                            add_pe_to_qkv=this_cfg.read_from_pixel.add_pe_to_qkv)                                    
        self.pixel_ffn2 = PixelFFN(self.embed_dim)

    def forward(
            self,
            pixel: torch.Tensor,
            pixel_pe: torch.Tensor,
            msk_value: torch.Tensor,
            mask_pe: torch.Tensor,
          need_weights: bool = False) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        # x: (bs*num_objects)*num_queries*embed_dim
        # pixel: bs*num_objects*C*H*W
        # query_pe: (bs*num_objects)*num_queries*embed_dim
        # pixel_pe: (bs*num_objects)*(H*W)*C
        # attn_mask: (bs*num_objects*num_heads)*num_queries*(H*W)
        b, n, c, h, w = pixel.shape
        # bs*num_objects*C*H*W -> (bs*num_objects)*(H*W)*C

        pixel_flat = pixel.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()
        msk_value = msk_value.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()

        ### First
        ### learnable parameter q, mask_value kv attention

        msk_value, p_weights = self.out_from_object(msk_value,
                                             pixel_flat,
                                             mask_pe,
                                             pixel_pe,
                                             attn_mask=None,
                                             need_weights=need_weights)
        msk_value = self.ffn(msk_value)


        pixel_semantic, p_weights = self.out_from_mask(pixel_flat,
                                             msk_value,
                                             pixel_pe,
                                             mask_pe,
                                             attn_mask=None,
                                             need_weights=need_weights)

        pixel_flat = self.pixel_ffn2(pixel, pixel_semantic)
        msk_value = msk_value.view(b, n, c, h, w)
        return pixel_flat, msk_value, p_weights, p_weights

class QueryTransformer(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        this_cfg = model_cfg.object_transformer
        self.value_dim = model_cfg.value_dim
        self.embed_dim = this_cfg.embed_dim
        self.num_heads = this_cfg.num_heads
        self.num_queries = this_cfg.num_queries

        self.pixel_pe_scale = model_cfg.pixel_pe_scale
        self.pixel_pe_temperature = model_cfg.pixel_pe_temperature
        self.pixel_init_proj = GConv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.pixel_emb_proj = GConv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.spatial_pe = PositionalEncoding(self.embed_dim,
                                             scale=self.pixel_pe_scale,
                                             temperature=self.pixel_pe_temperature,
                                             channel_last=False,
                                             transpose_output=True)
        self.mask_init_proj = GConv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.mask_emb_proj = GConv2d(self.embed_dim, self.embed_dim, kernel_size=1)
        self.spatial_mask_pe = PositionalEncoding(self.embed_dim,
                                                  scale=self.pixel_pe_scale,
                                                  temperature=self.pixel_pe_temperature,
                                                  channel_last=False,
                                                  transpose_output=True)
        # transformer blocks
        self.num_blocks = this_cfg.num_blocks
        self.blocks = nn.ModuleList(
            QueryTransformerBlock(model_cfg) for _ in range(self.num_blocks))
        # self.mask = nn.Conv2d(1,1,1)
        self.mask_pred = nn.ModuleList(
            nn.Sequential(nn.ReLU(), GConv2d(self.embed_dim, 1, kernel_size=1))
            for _ in range(self.num_blocks + 1))

        self.act = nn.ReLU(inplace=True)

    def forward(self,
                pixel: torch.Tensor,
                msk_value: torch.Tensor,

                selector: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> (torch.Tensor, Dict[str, torch.Tensor]):
        # pixel: B*num_objects*embed_dim*H*W
        # obj_summaries: B*num_objects*T*num_queries*embed_dim
        # msk_value : B * num_objects * CV * T * H * W

        bs, num_objects, _, H, W = pixel.shape

        msk_value = msk_value.permute(0, 1, 3, 2, 4, 5)
        T = msk_value.shape[2]
        msk_value = msk_value.sum(dim=2) / T


        # positional embeddings for pixel features
        pixel_init = self.pixel_init_proj(pixel)
        pixel_emb = self.pixel_emb_proj(pixel)
        pixel_pe = self.spatial_pe(pixel.flatten(0, 1))
        pixel_emb = pixel_emb.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()
        pixel_pe = pixel_pe.flatten(1, 2) + pixel_emb
        pixel = pixel_init

        mask_init = self.mask_init_proj(msk_value)
        mask_emb = self.mask_emb_proj(msk_value)
        mask_pe = self.spatial_mask_pe(msk_value.flatten(0, 1))
        mask_emb = mask_emb.flatten(3, 4).flatten(0, 1).transpose(1, 2).contiguous()
        mask_pe = mask_pe.flatten(1, 2) + mask_emb
        msk_value = mask_init


        # run the transformer
        aux_features = {'logits': []}

        # first aux output
        aux_logits = self.mask_pred[0](pixel).squeeze(2)
        attn_mask = self._get_aux_mask(aux_logits, selector)
        aux_features['logits'].append(aux_logits)

        for i in range(self.num_blocks):
            pixel, msk_value, q_weights, p_weights = self.blocks[i](    pixel,
                                                                        pixel_pe,
                                                                        msk_value,
                                                                        mask_pe,
                                                                        need_weights=need_weights)

            ### query update
            aux_logits = self.mask_pred[i + 1](pixel).squeeze(2)
            aux_features['logits'].append(aux_logits)
            msk_value = msk_value * aux_logits.unsqueeze(2).sigmoid()

        aux_features['q_weights'] = q_weights  # last layer only
        aux_features['p_weights'] = p_weights  # last layer only
        if self.training:
            # no need to save all heads
            aux_features['attn_mask'] = attn_mask.view(bs, num_objects, self.num_heads,
                                                       self.num_queries, H, W)[:, :, 0]

        return pixel + msk_value, aux_features

    def _get_aux_mask(self, logits: torch.Tensor, selector: torch.Tensor) -> torch.Tensor:
        # logits: batch_size*num_objects*H*W
        # selector: batch_size*num_objects*1*1
        # returns a mask of shape (batch_size*num_objects*num_heads)*num_queries*(H*W)
        # where True means the attention is blocked

        if selector is None:
            prob = logits.sigmoid()
        else:
            prob = logits.sigmoid() * selector
        logits = aggregate(prob, dim=1)

        is_foreground = (logits[:, 1:] >= logits.max(dim=1, keepdim=True)[0])
        foreground_mask = is_foreground.bool().flatten(start_dim=2)
        inv_foreground_mask = ~foreground_mask
        inv_background_mask = foreground_mask

        aux_foreground_mask = inv_foreground_mask.unsqueeze(2).unsqueeze(2).repeat(
            1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)
        aux_background_mask = inv_background_mask.unsqueeze(2).unsqueeze(2).repeat(
            1, 1, self.num_heads, self.num_queries // 2, 1).flatten(start_dim=0, end_dim=2)

        aux_mask = torch.cat([aux_foreground_mask, aux_background_mask], dim=1)

        aux_mask[torch.where(aux_mask.sum(-1) == aux_mask.shape[-1])] = False

        return aux_mask
