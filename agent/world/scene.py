import os
import pybullet as p
import numpy as np
import pybullet_data
import time
from abc import ABC, abstractmethod
from agent.utils import transform_utils

class BaseScene(ABC):
    def __init__(self, world, config, rng, test=False, validate=False):
        self._world = world
        self._rng = rng
        self._model_path = pybullet_data.getDataPath()
        self._validate = validate
        self._test = test

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class OnFloor(BaseScene):
    # 3D workspace for tote 1
    def reset(self):
        self._workspace_bounds = np.array([
            [0.38, 0.62],  # 3x2 rows: x,y,z cols: min,max
            [-0.22, 0.22],
            [0.00, 0.5]
        ])

        # load environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8)

class OnTable(BaseScene):
    """Tabletop settings"""
    def reset(self):
        raise BaseException

        self.plane_path = 'plane/plane.urdf'
        self.table_path = 'table/table.urdf'
        self._model_path = pybullet_data.getDataPath()
        plane_urdf = os.path.join("assets", self.plane_path)
        table_urdf = os.path.join("assets", self.table_path)
        self._world.add_model(plane_urdf, [0., 0., -1.], [0., 0., 0., 1.])
        self._world.add_model(table_urdf, [0., 0., -.82], [0., 0., 0., 1.])
        
        # Wait for the objects to rest
        self._world.run(1.)

class OnTote(BaseScene):
    """Tote settings"""
    def reset(self):
        raise BaseException
        