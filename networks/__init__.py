from importlib.util import find_spec
from pathlib import Path
import sys

# Direct inference entry points also need the MPViT registry used by training.
if find_spec("mmseg") is None:
    sys.path.append(str(Path(__file__).resolve().parents[1] / "_shims"))

from .resnet_encoder import ResnetEncoder
from .pose_decoder import PoseDecoder
#from .pose_cnn import PoseCNN
from .hr_decoder import DepthDecoder
from .mpvit import *
from .nets import DeepNet
from .auto_blur import AutoBlurModule
