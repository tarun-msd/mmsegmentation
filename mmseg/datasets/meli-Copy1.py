from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os

classes = ('background', 'foreground')
palette = [[0,0,0], [255,255,255]]

@DATASETS.register_module()
class MeliDataset(CustomDataset):
    CLASSES = classes
    PALETTE = palette
    def __init__(self, split, **kwargs):
                  super().__init__(img_suffix='.jpg', seg_map_suffix='.jpg', split=split, **kwargs)
                  assert os.path.exists(self.img_dir) and self.split is not None
