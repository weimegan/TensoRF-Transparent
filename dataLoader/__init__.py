from .llff import LLFFDataset
from .blender import BlenderDataset
from .nsvf import NSVF
from .tankstemple import TanksTempleDataset
from .dexnerfrealtable import DexNerfRealTable
from .your_own_data import YourOwnDataset
from .llff_features import LLFFFeatsDataset


dataset_dict = {'blender': BlenderDataset,
               'llff':LLFFDataset,
               'tankstemple':TanksTempleDataset,
               'nsvf':NSVF,
               "dexnerfrealtable": DexNerfRealTable,
                'own_data':YourOwnDataset,
                'llff_features': LLFFFeatsDataset}