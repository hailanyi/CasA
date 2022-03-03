from .detector3d_template import Detector3DTemplate
from .pv_rcnn import PVRCNN
from .voxel_rcnn import VoxelRCNN
from .CT3D import CT3D
from .CT3D_3CAT import CT3D_3CAT

__all__ = {
    'Detector3DTemplate': Detector3DTemplate,
    'PVRCNN': PVRCNN,
    'VoxelRCNN': VoxelRCNN,
    'CT3D': CT3D,
    "CT3D_3CAT": CT3D_3CAT,
}


def build_detector(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg, num_class=num_class, dataset=dataset
    )

    return model
