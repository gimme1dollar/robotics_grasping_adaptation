import os
import pybullet as p
import numpy as np
import pybullet_data

from abc import ABC, abstractmethod
from agent.utils import transform

class BaseScene(ABC):
    def __init__(self, world, config, rng, test=False, validate=False):
        self._world = world
        self._rng = rng
        self._model_path = pybullet_data.getDataPath()
        self._validate = validate
        self._test = test
        self.extent = config.get('extent', 0.1)
        self.max_objects = config.get('max_objects', 6)
        self.min_objects = config.get('min_objects', 1)
        self.num_objects = config.get('num_objects', 7)
        object_samplers = {'wooden_blocks': self._sample_wooden_blocks,
                           'random_urdfs': self._sample_random_objects}
        self._object_sampler = object_samplers[config['scene']['data_set']]
        print("dataset", config['scene']['data_set'])

    def _sample_wooden_blocks(self, n_objects):
        self._model_path = "assets/"
        object_names = ['circular_segment', 'cube',
                        'cuboid0', 'cuboid1', 'cylinder', 'triangle']
        selection = self._rng.choice(object_names, size=n_objects)
        paths = [os.path.join(self._model_path, 'objects',
                              name + '.urdf') for name in selection]
        return paths, 1.

    def _sample_random_objects(self, n_objects):
        if self._validate:
            self.object_range = np.arange(700, 850)
        elif self._test:
            self.object_range = np.arange(850, 1000)
        else: 
            self.object_range = 700
        # object_range = 900 if not self._test else np.arange(900, 1000)
        selection = self._rng.choice(self.object_range, size=n_objects)
        paths = [os.path.join(self._model_path, 'random_urdfs',
                            '{0:03d}/{0:03d}.urdf'.format(i)) for i in selection]
        return paths, 1.

    @abstractmethod
    def reset(self):
        raise NotImplementedError

class OnTable(BaseScene):
    """Tabletop settings with geometrically different objects."""
    def reset(self):
        self.table_path = 'table/table.urdf'
        self.plane_path = 'plane/plane.urdf'
        self._model_path = pybullet_data.getDataPath()
        plane_urdf = os.path.join("assets", self.plane_path)
        table_urdf = os.path.join("assets", self.table_path)
        self._world.add_model(plane_urdf, [0., 0., -1.], [0., 0., 0., 1.])
        self._world.add_model(table_urdf, [0., 0., -.82], [0., 0., 0., 1.])
        
        # Sample random objects
        n_objects = self._rng.randint(self.num_objects - 2, self.num_objects + 2)
        urdf_paths, scale = self._object_sampler(n_objects)

        # Spawn objects
        for path in urdf_paths:
            position = np.r_[self._rng.uniform(-self.extent, self.extent, 2), 0.1]
            orientation = transform.random_quaternion(self._rng.rand(3))
            self._world.add_model(path, position, orientation, scaling=scale)
            self._world.run(0.4)

        # Wait for the objects to rest
        self._world.run(1.)

class OnFloor(BaseScene):
    """Curriculum paper setup."""
    def reset(self):
        self.plane_path = 'plane/plane.urdf'
        plane_urdf = os.path.join("assets", self.plane_path)
        self._world.add_model(plane_urdf, [0., 0., -0.196], [0., 0., 0., 1.])
        # Sample random objects
        n_objects = self._rng.randint(self.num_objects - 2, self.num_objects + 2)
        urdf_paths, scale = self._object_sampler(n_objects)
        
        # Spawn objects
        for path in urdf_paths:
            position = np.r_[self._rng.uniform(-self.extent, self.extent, 2), 0.1]
            orientation = transform.random_quaternion(self._rng.rand(3))
            self._world.add_model(path, position, orientation, scaling=scale)
            self._world.run(0.4)

        # Wait for the objects to rest
        self._world.run(1.)


class OnTableWithBox(BaseScene):
    """Google Q-opt setup."""
    pass

