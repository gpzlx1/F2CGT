import torch
import os

package_path = os.path.dirname(os.path.abspath(os.path.expanduser(__file__)))
so_path = os.path.join(package_path, 'libpg.so')
torch.classes.load_library(so_path)

from .feature_cache import FeatureCacheServer