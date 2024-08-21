# ------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

import copy
from typing import Optional, List
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.models.utils import Transformer
from mmdet.models.utils import build_transformer
from .ms_deform_attn import *
from .ms_deform_attn_func import *

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         MultiScaleDeformableAttention,
                                         TransformerLayerSequence,
                                         build_transformer_layer_sequence)

from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)

@TRANSFORMER.register_module()
class RangeImageDeformableTransformer(Transformer):
    def __init__(self, num_feature_levels=4,**kwargs):
        super(RangeImageDeformableTransformer, self).__init__(**kwargs)
        self.embed_dims = self.encoder.embed_dims
        self.num_feature_levels = num_feature_levels
        self.init_layers()
        
        
    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

        xavier_uniform_(self.reference_points.weight.data, gain=1.0)
        constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None):
        """
        :param srcs   (list(Tensor)): Input features from different level. Each element has shape [layer, embed_dims, h, w]. This come from static map to be querid
        :param masks  (list(Tensor)): The key_padding_mask from different level used for encoder and decoder, each element has shape  [layer, h, w]. 
        :param pos_embeds (list(Tensor)): The positional encoding of feats from different level, has the shape [layer, embed_dims, h, w].
        :param query_embed (Tensor):  The query embedding for decoder, with shape [layer, embed_dims, h, w]. This come from dynamic map to query feature from static map
        """
        # prepare input for encoder
        layer_size = len(srcs)
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2) # layer, embed_dim, h, w -> layer, h * w, embed_dim 
            mask = mask.flatten(1) # layer, h, w -> layer, h * w
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # layer, embed_dim, h, w -> layer, h * w, embed_dim 
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1) # layer, h * w, embed_dim  + 1, 1, embed_dim -> layer, h * w, embed_dim 
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        # query_embed, tgt = torch.split(query_embed, c, dim=1)
        # query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
        # tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
        squeezed_query_embed = query_embed.permute(0, 2, 3, 1).reshape(layer_size, -1, self.embed_dims) # layer, embed_dims, h, w  -> layer, h * w, embed_dims
        reference_points = self.reference_points(squeezed_query_embed.reshape(-1, self.embed_dims)).sigmoid().reshape(layer_size, -1, 2)
        init_reference_out = reference_points

        # decoder
        hs, inter_references = self.decoder(squeezed_query_embed, reference_points, memory,
                                            spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten)
        hs = hs.permute(0, 2, 1).reshape(layer_size, self.embed_dims, query_embed.shape[2], query_embed.shape[3])

        inter_references_out = inter_references
        return hs, init_reference_out, inter_references_out

@TRANSFORMER_LAYER.register_module()
class RangeImageDeformableTransformerEncoderLayer(BaseModule):
    def __init__(self, embed_dims, d_model, n_levels, n_heads, n_points, dropout, d_ffn, activation, **kwargs):
        super(RangeImageDeformableTransformerEncoderLayer, self).__init__(**kwargs)
        self.embed_dims = embed_dims
        self.pre_norm = False

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        ''' 
        :param src(Tensor): Input features from different level.  [layer, embed_dims, h, w]. This come from static map to be querid
        :param pos(Tensor): Pose embeding, shape is  layer, h * w, embed_dims
        :param reference_points(Tensor): Default reference points, shape is layer, h * w, 2
        :param spatial_shapes: Difference shape for different layers , shape is  layer , 2
        :param level_start_index: start index for different layer , shape is one dimentional tensor size of layer
        :param padding_mask: (batch, h * w ), True for padding elements, False for non-padding elements
        '''
        
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class RangeImageDeformableTransformerEncoder(TransformerLayerSequence):
    def __init__(self, **kwargs):
        super(RangeImageDeformableTransformerEncoder, self).__init__(**kwargs)
        self.embed_dims = self.layers[0].embed_dims
        # self.layers = _get_clones(encoder_layer, num_layers)
        # self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output

@TRANSFORMER_LAYER.register_module()
class RangeImageDeformableTransformerDecoderLayer(BaseModule):
    def __init__(self, embed_dims, d_model, n_levels, n_heads, n_points, dropout, d_ffn, activation, **kwargs):
        super(RangeImageDeformableTransformerDecoderLayer, self).__init__( **kwargs)
        self.embed_dims = embed_dims
        self.pre_norm = False
        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, tgt, query_pos, reference_points, src, src_spatial_shapes, level_start_index, src_padding_mask=None):
        ''' 
        :param tgt(Tensor): From squeezed_query_embed(squeezed dynamic map) to be updated to contain all info, shape is  layer, h * w, embed_dims
        :param query_pos: Pose embeding, shape is  layer, h * w, embed_dims
        :param reference_points(Tensor): Optimized reference points obtained by linear transfom of  squeezed_query_embed, shape is layer, h * w, 2
        :param src (Tensor): From Encoder output, shape is layer, h * w, embed_dims 
        :param src_spatial_shapes: Difference shape for different layers , shape is  layer , 2
        :param level_start_index: start index for different layer , shape is one dimentional tensor size of layer
        :param src_padding_mask: (batch, h * w ), True for padding elements, False for non-padding elements
        '''
        # self attention  
        tgt2 = self.self_attn(self.with_pos_embed(tgt, query_pos), reference_points, tgt, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # cross attention
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class RangeImageDeformableTransformerDecoder(TransformerLayerSequence):
    def __init__(self, return_intermediate=False, **kwargs):
        super(RangeImageDeformableTransformerDecoder, self).__init__( **kwargs)
        self.bbox_embed = None
        self.class_embed = None
        # self.layers = _get_clones(decoder_layer, num_layers)
        # self.num_layers = num_layers
        self.return_intermediate = return_intermediate


    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None):
        '''
        :param tgt(Tensor):  From squeezed_query_embed(squeezed dynamic map) to be updated to contain all info, shape is  layer, h * w, embed_dims
        :param reference_points(Tensor): Optimized reference points obtained by linear transfom of  squeezed_query_embed, shape is layer, h * w, 2
        :param src (Tensor): From Encoder output, shape is layer, h * w, embed_dims 
        :param src_spatial_shapes(Tensor): Difference shape for different layers , shape is  layer , 2
        :param src_level_start_index(Tensor): start index for different layer , shape is one dimentional tensor size of layer
        :param src_valid_ratios(Tensor): ratio for diffence layer to conver reference points,  shape is layer, 1, 2
        :param query_pos(Tensor): Pose embeding, shape is  layer, h * w, embed_dims
        :param src_padding_mask(Tensor):  add mask to fit for different type, shape is  layer, h * w
        
        '''
        
        #reference point here should be majorly focused on motion objects
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 4:
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]
            output = layer(output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask)

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == 4:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:
                    assert reference_points.shape[-1] == 2
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")




