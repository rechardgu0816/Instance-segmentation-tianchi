import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (build_conv_layer, build_norm_layer, constant_init,
                      kaiming_init)
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm

from mmdet.ops import build_plugin_layer
from mmdet.utils import get_root_logger
from ..builder import BACKBONES
from ..utils import ResLayer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None):
        super(BasicBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        self.conv2 = build_conv_layer(
            conv_cfg, planes, planes, 3, padding=1, bias=False)
        self.add_module(self.norm2_name, norm2)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.norm2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 style='pytorch',
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 dcn=None,
                 plugins=None):
        """Bottleneck block for ResNet.
        If style is "pytorch", the stride-two layer is the 3x3 conv layer,
        if it is "caffe", the stride-two layer is the first 1x1 conv layer.
        """
        super(Bottleneck, self).__init__()
        assert style in ['pytorch', 'caffe']
        assert dcn is None or isinstance(dcn, dict)
        assert plugins is None or isinstance(plugins, list)
        if plugins is not None:
            allowed_position = ['after_conv1', 'after_conv2', 'after_conv3']
            assert all(p['position'] in allowed_position for p in plugins)

        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        self.dilation = dilation
        self.style = style
        self.with_cp = with_cp
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.dcn = dcn
        self.with_dcn = dcn is not None
        self.plugins = plugins
        self.with_plugins = plugins is not None

        if self.with_plugins:
            # collect plugins for conv1/conv2/conv3
            self.after_conv1_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv1'
            ]
            self.after_conv2_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv2'
            ]
            self.after_conv3_plugins = [
                plugin['cfg'] for plugin in plugins
                if plugin['position'] == 'after_conv3'
            ]

        if self.style == 'pytorch':
            self.conv1_stride = 1
            self.conv2_stride = stride
        else:
            self.conv1_stride = stride
            self.conv2_stride = 1

        self.norm1_name, norm1 = build_norm_layer(norm_cfg, planes, postfix=1)
        self.norm2_name, norm2 = build_norm_layer(norm_cfg, planes, postfix=2)
        self.norm3_name, norm3 = build_norm_layer(
            norm_cfg, planes * self.expansion, postfix=3)

        self.conv1 = build_conv_layer(
            conv_cfg,
            inplanes,
            planes,
            kernel_size=1,
            stride=self.conv1_stride,
            bias=False)
        self.add_module(self.norm1_name, norm1)
        fallback_on_stride = False
        if self.with_dcn:
            fallback_on_stride = dcn.pop('fallback_on_stride', False)
        if not self.with_dcn or fallback_on_stride:
            self.conv2 = build_conv_layer(
                conv_cfg,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)
        else:
            assert self.conv_cfg is None, 'conv_cfg must be None for DCN'
            self.conv2 = build_conv_layer(
                dcn,
                planes,
                planes,
                kernel_size=3,
                stride=self.conv2_stride,
                padding=dilation,
                dilation=dilation,
                bias=False)

        self.add_module(self.norm2_name, norm2)
        self.conv3 = build_conv_layer(
            conv_cfg,
            planes,
            planes * self.expansion,
            kernel_size=1,
            bias=False)
        self.add_module(self.norm3_name, norm3)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

        if self.with_plugins:
            self.after_conv1_plugin_names = self.make_block_plugins(
                planes, self.after_conv1_plugins)
            self.after_conv2_plugin_names = self.make_block_plugins(
                planes, self.after_conv2_plugins)
            self.after_conv3_plugin_names = self.make_block_plugins(
                planes * self.expansion, self.after_conv3_plugins)

    def make_block_plugins(self, in_channels, plugins):
        """ make plugins for block

        Args:
            in_channels (int): Input channels of plugin.
            plugins (list[dict]): List of plugins cfg to build.

        Returns:
            list[str]: List of the names of plugin.

        """
        assert isinstance(plugins, list)
        plugin_names = []
        for plugin in plugins:
            plugin = plugin.copy()
            name, layer = build_plugin_layer(
                plugin,
                in_channels=in_channels,
                postfix=plugin.pop('postfix', ''))
            assert not hasattr(self, name), f'duplicate plugin {name}'
            self.add_module(name, layer)
            plugin_names.append(name)
        return plugin_names

    def forward_plugin(self, x, plugin_names):
        out = x
        for name in plugin_names:
            out = getattr(self, name)(x)
        return out

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):

        def _inner_forward(x):
            identity = x

            out = self.conv1(x)
            out = self.norm1(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv1_plugin_names)

            out = self.conv2(out)
            out = self.norm2(out)
            out = self.relu(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv2_plugin_names)

            out = self.conv3(out)
            out = self.norm3(out)

            if self.with_plugins:
                out = self.forward_plugin(out, self.after_conv3_plugin_names)

            if self.downsample is not None:
                identity = self.downsample(x)

            out += identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out
def tb_make_res_layer(block,
                   inplanes,
                   planes,
                   blocks,
                   stride=1,
                   dilation=1,
                   style='pytorch',
                   with_cp=False,
                   conv_cfg=None,
                   norm_cfg=dict(type='BN'),
                   dcn=None,
                   gen_attention=None,
                   gen_attention_blocks=[]):
    downsample = None
    if stride != 1 or inplanes != planes * block.expansion:
        downsample = nn.Sequential(
            build_conv_layer(
                conv_cfg,
                inplanes,
                planes * block.expansion,
                kernel_size=1,
                stride=stride,
                bias=False),
            build_norm_layer(norm_cfg, planes * block.expansion)[1],
        )

    layers = []
    layers.append(
        block(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            dilation=dilation,
            downsample=downsample,
            style=style,
            with_cp=with_cp,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            dcn=dcn,
            gen_attention=gen_attention if
            (0 in gen_attention_blocks) else None))
    inplanes = planes * block.expansion
    for i in range(1, blocks):
        layers.append(
            block(
                inplanes=inplanes,
                planes=planes,
                stride=1,
                dilation=dilation,
                style=style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gcb=gcb,
                gen_attention=gen_attention if
                (i in gen_attention_blocks) else None))

    return nn.Sequential(*layers)



@BACKBONES.register_module()
class TB_ResNet(nn.Module):
    """ResNet backbone.

    Args:
        depth (int): Depth of resnet, from {18, 34, 50, 101, 152}.
        stem_channels (int): Number of stem channels. Default: 64.
        base_channels (int): Number of base channels of res layer. Default: 64.
        in_channels (int): Number of input image channels. Default: 3.
        num_stages (int): Resnet stages. Default: 4.
        strides (Sequence[int]): Strides of the first block of each stage.
        dilations (Sequence[int]): Dilation of each stage.
        out_indices (Sequence[int]): Output from which stages.
        style (str): `pytorch` or `caffe`. If set to "pytorch", the stride-two
            layer is the 3x3 conv layer, otherwise the stride-two layer is
            the first 1x1 conv layer.
        deep_stem (bool): Replace 7x7 conv in input stem with 3 3x3 conv
        avg_down (bool): Use AvgPool instead of stride conv when
            downsampling in the bottleneck.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        norm_cfg (dict): Dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only.
        plugins (list[dict]): List of plugins for stages, each dict contains:

            - cfg (dict, required): Cfg dict to build plugin.
            - position (str, required): Position inside block to insert
              plugin, options are 'after_conv1', 'after_conv2', 'after_conv3'.
            - stages (tuple[bool], optional): Stages to apply plugin, length
              should be same as 'num_stages'.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
        zero_init_residual (bool): Whether to use zero init for last norm layer
            in resblocks to let them behave as identity.

    Example:
        >>> from mmdet.models import ResNet
        >>> import torch
        >>> self = ResNet(depth=18)
        >>> self.eval()
        >>> inputs = torch.rand(1, 3, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 64, 8, 8)
        (1, 128, 4, 4)
        (1, 256, 2, 2)
        (1, 512, 1, 1)
    """

    arch_settings = {
        18: (BasicBlock, (2, 2, 2, 2)),
        34: (BasicBlock, (3, 4, 6, 3)),
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3))
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_stages=4,
                 strides=(1, 2, 2, 2),
                 dilations=(1, 1, 1, 1),
                 out_indices=(0, 1, 2, 3),
                 style='pytorch',
                 deep_stem=False,
                 avg_down=False,
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 norm_eval=True,
                 dcn=None,
                 stage_with_dcn=(False, False, False, False),
                 plugins=None,
                 with_cp=False,
                 zero_init_residual=True):
        super(TB_ResNet, self).__init__()
        if depth not in self.arch_settings:
            raise KeyError(f'invalid depth {depth} for resnet')
        self.depth = depth
        self.stem_channels = stem_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        assert num_stages >= 1 and num_stages <= 4
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == num_stages
        self.out_indices = out_indices
        assert max(out_indices) < num_stages
        self.style = style
        self.deep_stem = deep_stem
        self.avg_down = avg_down
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp
        self.norm_eval = norm_eval
        self.dcn = dcn
        self.stage_with_dcn = stage_with_dcn
        if dcn is not None:
            assert len(stage_with_dcn) == num_stages
        self.plugins = plugins
        self.zero_init_residual = zero_init_residual
        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks[:num_stages]
        self.inplanes = stem_channels

        self._make_stem_layer(in_channels, stem_channels)

        self.res_layers = []
        
        ### add db modules
        self.second_res_layers = []
        self.eb_second_conv1 = nn.Conv2d(
                256,
                64,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1, bias=False)
        self.eb_second_conv2 = nn.Conv2d(
            512,
            256,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1, bias=False)
        self.eb_second_conv3 = nn.Conv2d(
            1024,
            512,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1, bias=False)
        self.eb_second_conv4 = nn.Conv2d(
            2048,
            1024,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1, bias=False)
        #### add tb modules
        self.third_res_layers = []
        self.eb_third_conv1 = nn.Conv2d(
                256,
                64,
                kernel_size=1,
                stride=1,
                padding=0,
                dilation=1, bias=False)
        self.eb_third_conv2 = nn.Conv2d(
            512,
            256,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1, bias=False)
        self.eb_third_conv3 = nn.Conv2d(
            1024,
            512,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1, bias=False)
        self.eb_third_conv4 = nn.Conv2d(
            2048,
            1024,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1, bias=False)


        ######### add bn for db
        normalize = self.norm_cfg

        self.eb_second_bn1_name, eb_second_bn1 = build_norm_layer(normalize, 64, postfix='eb_second_bn1')
        self.add_module(self.eb_second_bn1_name, eb_second_bn1)

        self.eb_second_bn2_name, eb_second_bn2 = build_norm_layer(normalize, 256, postfix='eb_second_bn2')
        self.add_module(self.eb_second_bn2_name, eb_second_bn2)

        self.eb_second_bn3_name, eb_second_bn3 = build_norm_layer(normalize, 512, postfix='eb_second_bn3')
        self.add_module(self.eb_second_bn3_name, eb_second_bn3)

        self.eb_second_bn4_name, eb_second_bn4 = build_norm_layer(normalize, 1024, postfix='eb_second_bn4')
        self.add_module(self.eb_second_bn4_name, eb_second_bn4)
       
        ######### add bn for tb

        self.eb_third_bn1_name, eb_third_bn1 = build_norm_layer(normalize, 64, postfix='eb_third_bn1')
        self.add_module(self.eb_third_bn1_name, eb_third_bn1)

        self.eb_third_bn2_name, eb_third_bn2 = build_norm_layer(normalize, 256, postfix='eb_third_bn2')
        self.add_module(self.eb_third_bn2_name, eb_third_bn2)

        self.eb_third_bn3_name, eb_third_bn3 = build_norm_layer(normalize, 512, postfix='eb_third_bn3')
        self.add_module(self.eb_third_bn3_name, eb_third_bn3)

        self.eb_third_bn4_name, eb_third_bn4 = build_norm_layer(normalize, 1024, postfix='eb_third_bn4')
        self.add_module(self.eb_third_bn4_name, eb_third_bn4)

        ###### upsample in db
        self.eb_upsample = nn.Upsample(
            scale_factor=2, mode='nearest')

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            dcn = self.dcn if self.stage_with_dcn[i] else None
            if plugins is not None:
                stage_plugins = self.make_stage_plugins(plugins, i)
            else:
                stage_plugins = None
            planes = base_channels * 2**i
            res_layer = self.make_res_layer(
                block=self.block,
                inplanes=self.inplanes,
                planes=planes,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                avg_down=self.avg_down,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                plugins=stage_plugins)
            second_res_layer = tb_make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gen_attention=gen_attention,
                gen_attention_blocks=stage_with_gen_attention[i])
            third_res_layer = tb_make_res_layer(
                self.block,
                self.inplanes,
                planes,
                num_blocks,
                stride=stride,
                dilation=dilation,
                style=self.style,
                with_cp=with_cp,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                dcn=dcn,
                gen_attention=gen_attention,
                gen_attention_blocks=stage_with_gen_attention[i])
            
            self.inplanes = planes * self.block.expansion
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, res_layer)
            self.res_layers.append(layer_name)
            #### add dual backbone
            second_layer_name = 'second_layer{}'.format(i + 1)
            self.add_module(second_layer_name, second_res_layer)
            self.second_res_layers.append(second_layer_name)
            
            #### add triple backbone
            third_layer_name = 'third_layer{}'.format(i + 1)
            self.add_module(third_layer_name, third_res_layer)
            self.third_res_layers.append(third_layer_name)
                                  
        self._freeze_stages()

        self.feat_dim = self.block.expansion * base_channels * 2**(
            len(self.stage_blocks) - 1)

    def make_stage_plugins(self, plugins, stage_idx):
        """ make plugins for ResNet 'stage_idx'th stage .

        Currently we support to insert 'context_block',
        'empirical_attention_block', 'nonlocal_block' into the backbone like
        ResNet/ResNeXt. They could be inserted after conv1/conv2/conv3 of
        Bottleneck.
        An example of plugins format could be:

        >>> plugins=[
        ...     dict(cfg=dict(type='xxx', arg1='xxx'),
        ...          stages=(False, True, True, True),
        ...          position='after_conv2'),
        ...     dict(cfg=dict(type='yyy'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='1'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3'),
        ...     dict(cfg=dict(type='zzz', postfix='2'),
        ...          stages=(True, True, True, True),
        ...          position='after_conv3')
        ... ]
        >>> self = ResNet(depth=18)
        >>> stage_plugins = self.make_stage_plugins(plugins, 0)
        >>> assert len(stage_plugins) == 3

        Suppose 'stage_idx=0', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->conv3->yyy->zzz1->zzz2

        Suppose 'stage_idx=1', the structure of blocks in the stage would be:

        .. code-block:: none

            conv1-> conv2->xxx->conv3->yyy->zzz1->zzz2

        If stages is missing, the plugin would be applied to all stages.

        Args:
            plugins (list[dict]): List of plugins cfg to build. The postfix is
                required if multiple same type plugins are inserted.
            stage_idx (int): Index of stage to build

        Returns:
            list[dict]: Plugins for current stage
        """
        stage_plugins = []
        for plugin in plugins:
            plugin = plugin.copy()
            stages = plugin.pop('stages', None)
            assert stages is None or len(stages) == self.num_stages
            # whether to insert plugin into current stage
            if stages is None or stages[stage_idx]:
                stage_plugins.append(plugin)

        return stage_plugins

    def make_res_layer(self, **kwargs):
        return ResLayer(**kwargs)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_stem_layer(self, in_channels, stem_channels):
        if self.deep_stem:
            self.stem = nn.Sequential(
                build_conv_layer(
                    self.conv_cfg,
                    in_channels,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels // 2,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels // 2)[1],
                nn.ReLU(inplace=True),
                build_conv_layer(
                    self.conv_cfg,
                    stem_channels // 2,
                    stem_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False),
                build_norm_layer(self.norm_cfg, stem_channels)[1],
                nn.ReLU(inplace=True))
        else:
            self.conv1 = build_conv_layer(
                self.conv_cfg,
                in_channels,
                stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False)
            self.norm1_name, norm1 = build_norm_layer(
                self.norm_cfg, stem_channels, postfix=1)
            self.add_module(self.norm1_name, norm1)
            self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem:
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            else:
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(
                            m.conv2, 'conv_offset'):
                        constant_init(m.conv2.conv_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError('pretrained must be a str or None')
     


    def get_cur_norm(self, key, i):
        if key=='second' and i==0:return getattr(self,self.eb_second_bn1_name)
        if key=='second' and i == 1: return getattr(self, self.eb_second_bn2_name)
        if key=='second' and i == 2: return getattr(self, self.eb_second_bn3_name)
        if key=='second' and i == 3: return getattr(self, self.eb_second_bn4_name)

        if key=='third' and i==0:return getattr(self,self.eb_third_bn1_name)
        if key=='third' and i == 1: return getattr(self, self.eb_third_bn2_name)
        if key=='third' and i == 2: return getattr(self, self.eb_third_bn3_name)
        if key=='third' and i == 3: return getattr(self, self.eb_third_bn4_name)

        if key=='four' and i==0:return getattr(self,self.eb_four_bn1_name)
        if key=='four' and i == 1: return getattr(self, self.eb_four_bn2_name)
        if key=='four' and i == 2: return getattr(self, self.eb_four_bn3_name)
        if key=='four' and i == 3: return getattr(self, self.eb_four_bn4_name)

        assert 1>5

    def get_cur_conv(self, key, i):
        if key=='second' and i == 0: return self.eb_second_conv1
        if key=='second' and i == 1: return self.eb_second_conv2
        if key=='second' and i == 2: return self.eb_second_conv3
        if key=='second' and i == 3: return self.eb_second_conv4

        if key=='third' and i == 0: return self.eb_third_conv1
        if key=='third' and i == 1: return self.eb_third_conv2
        if key=='third' and i == 2: return self.eb_third_conv3
        if key=='third' and i == 3: return self.eb_third_conv4

        if key=='four' and i == 0: return self.eb_four_conv1
        if key=='four' and i == 1: return self.eb_four_conv2
        if key=='four' and i == 2: return self.eb_four_conv3
        if key=='four' and i == 3: return self.eb_four_conv4

        assert 1>5

    def forward(self, x):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
            x = self.norm1(x)
            x = self.relu(x)
        x = self.maxpool(x)
        second_x = x
        third_x = x

        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            second_layer_name = self.second_res_layers[i]
            second_res_layer = getattr(self, second_layer_name)
            third_layer_name = self.third_res_layers[i]
            third_res_layer = getattr(self, third_layer_name)

            third_x = third_res_layer(third_x)
            cur_third_norm = self.get_cur_norm('third',i)
            cur_third_conv = self.get_cur_conv('third',i)
            tmp = cur_third_conv(third_x)
            tmp = cur_third_norm(tmp)
            if i!=0:tmp = self.eb_upsample(tmp)
            second_x = second_x + tmp

            second_x = second_res_layer(second_x)
            cur_second_norm = self.get_cur_norm('second',i)
            cur_second_conv = self.get_cur_conv('second',i)
            tmp = cur_second_conv(second_x)
            tmp = cur_second_norm(tmp)
            if i!=0:tmp = self.eb_upsample(tmp)
            x = x + tmp

            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)

    def train(self, mode=True):
        super(TB_ResNet, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

