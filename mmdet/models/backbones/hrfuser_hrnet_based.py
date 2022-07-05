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
from .hrformer import HRFormer, HRFormerBlock, CrossFFN
from .hrnet import Bottleneck, HRModule, HRNet
from .hrfuser_hrformer_based import HRFuserFusionBlock


@BACKBONES.register_module()
class HRFuserHRNetBased(HRNet):
    """HRFuser backbone.
    """
    blocks_dict = {'BASIC': BasicBlock, 'BOTTLENECK': Bottleneck, 
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