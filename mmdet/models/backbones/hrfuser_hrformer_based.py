import math
import warnings

import torch
import torch.nn as nn
from mmcv.cnn import (build_activation_layer, build_conv_layer,
                      build_norm_layer, constant_init, trunc_normal_init)
from mmcv.cnn.bricks.transformer import build_dropout
from mmcv.runner import BaseModule, _load_checkpoint
from mmcv.utils.parrots_wrapper import _BatchNorm
from torch.nn import functional as F
from torch.nn.functional import dropout, linear, pad, softmax

from ...utils import get_root_logger
from ..builder import BACKBONES
from ..utils import nchw_to_nlc, nlc_to_nchw
from .resnet import BasicBlock, Bottleneck
from .hrnet import HRModule, HRNet
from .hrformer import HRFormer, HRFormerBlock, CrossFFN

class WindowMCA(BaseModule):
    """Window based multi-head cross-attention (W-MCA) module with relative
    position bias.
    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float): Dropout ratio of output. Default: 0.
        with_rpe (bool): If True, use relative position bias.
            Default: True.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dim,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 kdim=None,
                 vdim=None,
                 with_rpe=True,
                 init_cfg=None):

        super().__init__(init_cfg=init_cfg)
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.Wh, self.Ww = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dim = embed_dim // num_heads
        self.scale = qk_scale or head_embed_dim**-0.5

        self.with_rpe = with_rpe
        if self.with_rpe:
            # define a parameter table of relative position bias
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * self.Wh - 1) * (2 * self.Ww - 1),
                            num_heads))  # (2*Wh-1) * (2*Ww-1), nH

            # pairwise relative position for each token inside the window
            coords_h = torch.arange(self.Wh)
            coords_w = torch.arange(self.Ww)
            # [2, Wh, Ww]
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
            # [2, Wh*Ww]
            coords_flatten = torch.flatten(coords, 1)
            # [2, Wh*Ww, Wh*Ww]
            relative_coords = coords_flatten[:, :, None] - \
                coords_flatten[:, None, :]
            # [Wh*Ww, Wh*Ww, 2]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            # shift the values to make them start from 0
            relative_coords[:, :, 0] += self.Wh - 1
            relative_coords[:, :, 1] += self.Ww - 1
            relative_coords[:, :, 0] *= 2 * self.Ww - 1
            # [Wh*Ww, Wh*Ww]
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer('relative_position_index',
                                 relative_position_index)

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=qkv_bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=qkv_bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)

    def init_weights(self):
        super(WindowMCA, self).init_weights()
        if self.with_rpe:
            trunc_normal_init(self.relative_position_bias_table, std=0.02)

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query (tensor): primary input features with shape of (B*num_windows, N, C)
            key (tensor): secondary input features with shape of (B*num_windows, N, C)
            value (tensor): secondary input features with shape of (B*num_windows, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = query.shape 

        # allow MHA to have different sizes for the feature dimension
        assert key.size(0) == value.size(0) and key.size(1) == value.size(1)

        # whether or not use the original query/key/value
        q = self.q_proj(query).reshape(B, N, self.num_heads,
                                  C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(key).reshape(B, N, self.num_heads,
                                  C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(value).reshape(B, N, self.num_heads,
                                  C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.with_rpe:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)].view(
                    self.Wh * self.Ww, self.Wh * self.Ww, -1)  # Wh*Ww,Wh*Ww,nH
            relative_position_bias = relative_position_bias.permute(
                2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.out_proj(x)
        x = self.proj_drop(x)
        return x

class MultiWindowCrossAttention(BaseModule):
    """Multi-window Cross Attention (MWCA) module with relative position bias.
    Args:
        embed_dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int] | int): The height and width of the window.
        qkv_bias (bool):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float): Dropout ratio of output. Default: 0.
        with_rpe (bool): If True, use relative position bias.
            Default: True.
        with_pad_mask (bool): If True, mask out the padded tokens in
            the attention process. Default: False.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 window_size=7,
                 with_pad_mask=False,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        if isinstance(window_size, int):
            window_size = (window_size, window_size)
        self.window_size = window_size
        self.with_pad_mask = with_pad_mask
        self.attn = WindowMCA(
            window_size=self.window_size,
            init_cfg=init_cfg,
            **kwargs)

    def forward(self, x, y, H, W, **kwargs):
        """Forward function.
        Args:
            x: (torch.Tensor): The input tensor with shape [B, N, C].
            y: (torch.Tensor): The second modality input tensor with shape [B, N, C].
            H: (int): The height of the original 4D feature map.
            W: (int): The width of the original 4D feature map.
            **kwargs: Other arguments input to the forward function
                of `WindowMSA`
        Returns:
            torch.Tensor: The output tensor with shape [B, N, C]
        """
        assert x.shape == y.shape # Assuming symetrical inputs
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        y = y.view(B, H, W, C)
        Wh, Ww = self.window_size

        # center-pad the feature on H and W axes
        pad_h = math.ceil(H / Wh) * Wh - H
        pad_w = math.ceil(W / Ww) * Ww - W
        x = pad(x, (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2))
        y = pad(y, (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2))

        # permute
        x = x.view(B, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, C)
        x = x.permute(0, 1, 3, 2, 4, 5)
        x = x.reshape(-1, Wh * Ww, C)  # (B*num_window, Wh*Ww, C)
        y = y.view(B, math.ceil(H / Wh), Wh, math.ceil(W / Ww), Ww, C)
        y = y.permute(0, 1, 3, 2, 4, 5)
        y = y.reshape(-1, Wh * Ww, C)

        # attention
        if self.with_pad_mask and pad_h > 0 and pad_w > 0:
            pad_mask = x.new_zeros(1, H, W, 1)
            pad_mask = pad(
                pad_mask, [
                    0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2,
                    pad_h - pad_h // 2
                ],
                value=-float('inf'))
            pad_mask = pad_mask.view(1, math.ceil(H / Wh), Wh,
                                     math.ceil(W / Ww), Ww, 1)
            pad_mask = pad_mask.permute(0, 1, 3, 2, 4, 5)
            pad_mask = pad_mask.reshape(-1, Wh * Ww)
            pad_mask = pad_mask[:, None, :].expand([-1, Wh * Ww, -1])
            out = self.attn(x, y, y, pad_mask, **kwargs)
        else:
            out = self.attn(x, y, y, **kwargs)

        # reverse permutation
        out = out.reshape(B, math.ceil(H / Wh), math.ceil(W / Ww), Wh, Ww, C)
        out = out.permute(0, 1, 3, 2, 4, 5)
        out = out.reshape(B, H + pad_h, W + pad_w, C)

        # de-pad
        out = out[:, pad_h // 2:H + pad_h // 2, pad_w // 2:W + pad_w // 2]
        return out.reshape(B, N, C)

class HRFuserFusionBlock(BaseModule):

    expansion = 1

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4,
                 drop_path=0.0,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='SyncBN'),
                 transformer_norm_cfg=dict(type='LN', eps=1e-6),
                 with_cp=False,
                 init_cfg=None,
                 num_fused_modalities=2,
                 **kwargs):
        super(HRFuserFusionBlock, self).__init__(init_cfg=init_cfg)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.with_cp = with_cp
        self.num_fused_modalities = num_fused_modalities

        norm1_list = []
        norm2_list = []
        attn_list = []
        for i in range(self.num_fused_modalities):
            norm1_list.append(build_norm_layer(transformer_norm_cfg, in_channels)[1])
            norm2_list.append(build_norm_layer(transformer_norm_cfg, out_channels)[1])
            attn_list.append(MultiWindowCrossAttention(
                embed_dim=in_channels,
                num_heads=self.num_heads,
                window_size=self.window_size,
                init_cfg=None,
                **kwargs))
        self.norm1 = nn.ModuleList(norm1_list)
        self.norm2 = nn.ModuleList(norm2_list)
        self.attn = nn.ModuleList(attn_list)

        self.norm3 = build_norm_layer(transformer_norm_cfg, out_channels)[1]
        self.ffn = CrossFFN(
            in_channels=in_channels,
            hidden_channels=int(in_channels * self.mlp_ratio),
            out_channels=out_channels,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            dw_act_cfg=act_cfg,
            init_cfg=None)

        self.drop_path = build_dropout(
            dict(type='DropPath',
                drop_prob=drop_path)) if drop_path > 0.0 else nn.Identity()

    def _inner_forward(self, x, y):
        B, C, H, W = x.size()
        # Attention
        x = nchw_to_nlc(x)
        x_tmp = torch.empty_like(x).copy_(x) # Required for the input to the attn layer to stay the same in the loops
        for i in range(self.num_fused_modalities):
            z = y[i]
            z = nchw_to_nlc(z)
            x = x + z + self.drop_path(self.attn[i](self.norm1[i](x_tmp), self.norm2[i](z), H, W)) # Norm to 0,1 ->         
        # FFN
        x = x + self.drop_path(self.ffn(self.norm3(x), H, W))
        x = nlc_to_nchw(x, (H, W))
        return x

    def forward(self, x, y):
        """Forward function."""
        if self.with_cp and x.requires_grad:
            raise Exception('with_cp is currently not possible with CA Fusion module')
            # out = cp.checkpoint(self._inner_forward, x, y)
        else:
            out = self._inner_forward(x, y)
        return out



@BACKBONES.register_module()
class HRFuserHRFormerBased(HRFormer):
    """HRFuser backbone.
    """
    blocks_dict = {'BOTTLENECK': Bottleneck, 'HRFORMER': HRFormerBlock,
        'CA': HRFuserFusionBlock, 'MWCA': HRFuserFusionBlock}

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 transformer_norm_cfg=dict(type='LN', eps=1e-6),
                 norm_eval=False,
                 with_cp=False,
                 drop_path_rate=0.,
                 zero_init_residual=False,
                 multiscale_output=True,
                 pretrained=None,
                 init_cfg=None,
                 num_fused_modalities=2,
                 mod_in_channels=[3,3]):
        super().__init__(
            extra=extra,
            in_channels=in_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            norm_eval=norm_eval,
            with_cp=with_cp,
            zero_init_residual=zero_init_residual,
            multiscale_output=multiscale_output,
            pretrained=pretrained,
            init_cfg=init_cfg)        

        cfg = self.extra
        self.num_fused_modalities = num_fused_modalities
        self.transformer_norm_cfg = transformer_norm_cfg
        self.pre_neck_fusion = True if self.extra['LidarStageD'] else False

        self.extra['LidarStageB']['drop_path_rates'] = self.extra['stage2']['drop_path_rates']
        self.extra['LidarStageC']['drop_path_rates'] = self.extra['stage3']['drop_path_rates']
        if self.pre_neck_fusion:
            self.extra['LidarStageD']['drop_path_rates'] = self.extra['stage4']['drop_path_rates']
            print('Pre Neck Fusion Activated')

        conv_a = []
        norm_a = []
        conv_b = []
        norm_b = []
        for i in range(self.num_fused_modalities):
            conv_a.append(build_conv_layer(
                self.conv_cfg,
                mod_in_channels[i],
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False))
            norm_a.append(build_norm_layer(self.norm_cfg, 64)[1])
            conv_b.append(build_conv_layer(
                self.conv_cfg,
                64,
                64,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False))
            norm_b.append(build_norm_layer(self.norm_cfg, 64)[1])
        self.conv_a = nn.ModuleList(conv_a)
        self.norm_a = nn.ModuleList(norm_a)
        self.conv_b = nn.ModuleList(conv_b)
        self.norm_b = nn.ModuleList(norm_b)

        # Stage A
        self.stage_a_cfg = cfg['LidarStageA']
        num_channels = self.stage_a_cfg['num_channels'][0]
        block = self.blocks_dict[self.stage_a_cfg['block']]
        num_blocks = self.stage_a_cfg['num_blocks'][0]
        stage_a_out_channels = [[num_channels * block.expansion] for i in range(self.num_fused_modalities)]
        modalities = []
        for i in range(self.num_fused_modalities):
            modalities.append(self._make_layer(block, 64, num_channels, num_blocks))
        self.layer_a = nn.ModuleList(modalities)

        # Pre stage 2 transition and fusion
        self.fusion_a_cfg = cfg['ModFusionA']
        num_channels = self.fusion_a_cfg['num_channels']
        block = self.blocks_dict[self.fusion_a_cfg['block']]
        num_channels = [channel * block.expansion for channel in num_channels]
            # stimmen die num_channels fuer die transition? oder muss ich stage_2_cfg nehmen?
        self.transition_a = self._make_mod_transition_layer(stage_a_out_channels, num_channels)
        self.fusion_a = self._make_multimodal_fusion(
            block, self.fusion_a_cfg, num_channels)

        # Stage B
        self.stage_b_cfg = cfg['LidarStageB']
        num_channels = self.stage_b_cfg['num_channels']
        block = self.blocks_dict[self.stage_b_cfg['block']]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.stage_b, pre_stage_channels = self._make_mod_stage(self.stage_b_cfg, num_channels)

        # Pre stage 3 transition and fusion
        self.fusion_b_cfg = cfg['ModFusionB']
        num_channels = self.fusion_b_cfg['num_channels']
        block = self.blocks_dict[self.fusion_b_cfg['block']]
        self.transition_b = self._make_mod_transition_layer(pre_stage_channels, num_channels)
        self.fusion_b = self._make_multimodal_fusion(
            block, self.fusion_b_cfg, num_channels)
        
        # Stage C
        self.stage_c_cfg = cfg['LidarStageC']
        num_channels = self.stage_c_cfg['num_channels']
        block = self.blocks_dict[self.stage_c_cfg['block']]
        num_channels = [channel * block.expansion for channel in num_channels]
        self.stage_c, pre_stage_channels = self._make_mod_stage(self.stage_c_cfg, num_channels)        

        # Pre stage 4 transition and fusion
        self.fusion_c_cfg = cfg['ModFusionC']
        num_channels = self.fusion_c_cfg['num_channels']
        block = self.blocks_dict[self.fusion_c_cfg['block']]
        self.transition_c = self._make_mod_transition_layer(pre_stage_channels, num_channels)
        self.fusion_c = self._make_multimodal_fusion(
            block, self.fusion_c_cfg, num_channels)
        
        if self.pre_neck_fusion:
            # Stage D
            self.stage_d_cfg = cfg['LidarStageD']
            num_channels = self.stage_d_cfg['num_channels']
            block = self.blocks_dict[self.stage_d_cfg['block']]
            num_channels = [channel * block.expansion for channel in num_channels]
            self.stage_d, pre_stage_channels = self._make_mod_stage(self.stage_d_cfg, num_channels)        

            # Pre stage Neck transition and fusion
            self.fusion_d_cfg = cfg['ModFusionD']
            num_channels = self.fusion_d_cfg['num_channels']
            block = self.blocks_dict[self.fusion_d_cfg['block']]
            self.transition_d = self._make_mod_transition_layer(pre_stage_channels, num_channels)
            self.fusion_d = self._make_multimodal_fusion(
                block, self.fusion_d_cfg, num_channels)

    def _make_mod_stage(self, layer_config, in_channels):
        pre_stage_channels = []
        modalities = []
        for _ in range(self.num_fused_modalities):
            tmp_stage, tmp_channels = self._make_stage(
            layer_config, in_channels)
            modalities.append(tmp_stage)
            pre_stage_channels.append(tmp_channels)
        return nn.ModuleList(modalities), pre_stage_channels

    def _make_mod_transition_layer(self, pre_stage_channels, num_channels):
        modalities = []
        for num_mod in range(self.num_fused_modalities):
            modalities.append(self._make_transition_layer(pre_stage_channels[num_mod],
                                                       num_channels))
        return nn.ModuleList(modalities)

    def _make_multimodal_fusion(self,
                    block,
                    layer_config,
                    num_inchannels):
        """Make each stage."""
        num_branches = layer_config['num_branches']
        num_channels = layer_config['num_channels']
        num_heads = layer_config['num_heads']
        num_window_sizes = layer_config['window_sizes']
        num_mlp_ratios = layer_config['mlp_ratios']
        drop_path = layer_config['drop_path']

        pre_branches = []
        
        for branch_index in range(num_branches):
            if layer_config['block'] == 'CA' or layer_config['block'] == 'MWCA':
                pre_branches.append(
                    block(
                            num_inchannels[branch_index],
                            num_channels[branch_index],
                            num_heads=num_heads[branch_index],
                            window_size=num_window_sizes[branch_index],
                            mlp_ratio=num_mlp_ratios[branch_index],
                            drop_path=drop_path,
                            norm_cfg=self.norm_cfg,
                            transformer_norm_cfg=self.transformer_norm_cfg,
                            init_cfg=None,                        
                            num_fused_modalities=self.num_fused_modalities,
                            proj_drop_rate=layer_config['proj_drop_rate']))
            else:
                raise Exception('Not valid fusion block')
                    

        return nn.ModuleList(pre_branches)

    def forward(self, x, x_mod):
        """Forward function."""
        if not self.num_fused_modalities == len(x_mod):
            raise Exception('num_fused_modalities does not fit the given input length')

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.relu(x)
        x = self.layer1(x)

        n_list = []
        for k in range(self.num_fused_modalities):
            x_mod[k] = self.conv_a[k](x_mod[k])
            x_mod[k] = self.norm_a[k](x_mod[k])
            x_mod[k] = self.relu(x_mod[k])
            x_mod[k] = self.conv_b[k](x_mod[k])
            x_mod[k] = self.norm_b[k](x_mod[k])
            x_mod[k] = self.relu(x_mod[k])
            x_mod[k] = self.layer_a[k](x_mod[k])
            n_list.append([x_mod[k]])        

        # Stage 2 & B
        x_list = []
        m_list = []
        for i in range(self.stage2_cfg['num_branches']):
            if self.transition1[i][0] is not None:
                x_tmp = self.transition1[i][0](x)
            else:
                x_tmp = x
            m_tmp = []
            for k in range(self.num_fused_modalities):
                if self.transition_a[k][i] is not None:
                    m_tmp.append(self.transition_a[k][i](n_list[k][0]))
                else:
                    m_tmp.append(n_list[k][0])
            m_list.append(m_tmp)
            x_list.append(
                self.fusion_a[i](x_tmp, m_tmp))
        y_list = self.stage2(x_list)
        for k in range(self.num_fused_modalities):
            n_list[k] = self.stage_b[k]([m_list[0][k]])

        # Stage 3 & C fusion
        x_list = []
        m_list = []
        for i in range(self.stage3_cfg['num_branches']):
            if self.transition2[i] is not None:
                x_tmp = self.transition2[i](y_list[-1])
            else:
                x_tmp = y_list[i]
            m_tmp = []
            for k in range(self.num_fused_modalities):
                if self.transition_b[k][i] is not None:
                    m_tmp.append(self.transition_b[k][i](n_list[k][0]))
                else:
                    m_tmp.append(n_list[k][0])
            m_list.append(m_tmp)
            x_list.append(
                self.fusion_b[i](x_tmp, m_tmp))
        y_list = self.stage3(x_list)
        for k in range(self.num_fused_modalities):
            n_list[k] = self.stage_c[k]([m_list[0][k]])

        # Pre Stage 4 fusion
        x_list = []
        if self.pre_neck_fusion:
            m_list = []
        for i in range(self.stage4_cfg['num_branches']):
            if self.transition3[i] is not None:
                x_tmp = self.transition3[i](y_list[-1])
            else:
                x_tmp = y_list[i]
            m_tmp = []
            for k in range(self.num_fused_modalities):
                if self.transition_c[k][i] is not None:
                    m_tmp.append(self.transition_c[k][i](n_list[k][0]))
                else:
                    m_tmp.append(n_list[k][0])
            if self.pre_neck_fusion:
                m_list.append(m_tmp)
            x_list.append(
                self.fusion_c[i](x_tmp, m_tmp))
        y_list = self.stage4(x_list)

        # Mod stage D & pre neck fusion
        if self.pre_neck_fusion:
            for k in range(self.num_fused_modalities):
                n_list[k] = self.stage_d[k]([m_list[0][k]])            
            x_list = []
            for i in range(self.stage4_cfg['num_branches']):
                x_tmp = y_list[i]
                m_tmp = []
                for k in range(self.num_fused_modalities):
                    if self.transition_d[k][i] is not None:
                        m_tmp.append(self.transition_d[k][i](n_list[k][0]))
                    else:
                        m_tmp.append(n_list[k][0])
                x_list.append(
                    self.fusion_d[i](x_tmp, m_tmp))
            for i in range(len(x_list)):
                y_list[i] = self.relu(x_list[i])
        

        return y_list