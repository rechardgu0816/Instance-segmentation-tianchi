from .hourglass import HourglassNet
from .hrnet import HRNet
from .regnet import RegNet
from .res2net import Res2Net
from .resnet import ResNet, ResNetV1d
from .resnext import ResNeXt
from .ssd_vgg import SSDVGG
from .detectors_resnet import DetectoRS_ResNet
from .detectors_resnext import DetectoRS_ResNeXt
#from .db_resnet import DB_ResNet
#from .db_resnext import DB_ResNeXt
#from .tb_resnet import TB_ResNet
#from .tb_resnext import TB_ResNeXt
from .senet import SENet
from .resnest import ResNeSt
__all__ = [
    'RegNet', 'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'Res2Net',
    'HourglassNet','SENet','ResNeSt','DetectoRS_ResNet', 'DetectoRS_ResNeXt'
]
