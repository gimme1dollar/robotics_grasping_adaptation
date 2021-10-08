import os
import pybullet as p
import numpy as np
import pybullet_data
import time
from abc import ABC, abstractmethod
from agent.common import transform_utils

class BaseScene(ABC):
    def __init__(self, world, config, rng, test=False, validate=False):
        self._world = world
        self._rng = rng
        self._model_path = pybullet_data.getDataPath()
        self._validate = validate
        self._test = test
        self.extent = config.get('extent', 0.1)
        self.max_objects = config.get('max_objects', 5)
        self.min_objects = config.get('min_objects', 5)
        object_samplers = {'wooden_blocks': self._sample_wooden_blocks,
                           'random_urdfs': self._sample_random_objects}
        self._object_sampler = object_samplers[config['scene']['data_set']]
        print("dataset", config['scene']['data_set'])

    def _sample_wooden_blocks(self, n_objects):
        self._model_path = "assets/"
        object_names = ['circular_segment', 'cube',
                        'cuboid0', 'cuboid1', 'cylinder', 'triangle']
        selection = self._rng.choice(object_names, size=n_objects)
        paths = [os.path.join(self._model_path, 'wooden_blocks',
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
        tray_path = os.path.join(self._model_path, 'tray/tray.urdf')
        plane_urdf = os.path.join("assets", self.plane_path)
        table_urdf = os.path.join("assets", self.table_path)
        self._world.add_model(plane_urdf, [0., 0., -1.], [0., 0., 0., 1.])
        self._world.add_model(table_urdf, [0., 0., -.82], [0., 0., 0., 1.])
        #self._world.add_model(tray_path, [0, 0.075, -0.19],
        #                      [0.0, 0.0, 1.0, 0.0], scaling=1.2)

        # Sample random objects
        n_objects = self._rng.randint(self.min_objects, self.max_objects + 1)
        urdf_paths, scale = self._object_sampler(n_objects)

        # Spawn objects
        for path in urdf_paths:
            position = np.r_[self._rng.uniform(-self.extent, self.extent, 2), 0.1]
            orientation = transform_utils.random_quaternion(self._rng.rand(3))
            self._world.add_model(path, position, orientation, scaling=scale)
            self._world.run(0.4)

        # Wait for the objects to rest
        self._world.run(1.)

class OnFloor(BaseScene):
    """Curriculum paper setup."""
    def reset(self):
        self.plane_path = 'plane.urdf'
        plane_urdf = os.path.join("assets", self.plane_path)
        self._world.add_model(plane_urdf, [0., 0., -0.196], [0., 0., 0., 1.])
        # Sample random objects
        n_objects = self._rng.randint(self.min_objects, self.max_objects + 1)
        urdf_paths, scale = self._object_sampler(n_objects)
        
        # Spawn objects
        for path in urdf_paths:
            position = np.r_[self._rng.uniform(-self.extent, self.extent, 2), 0.1]
            orientation = transform_utils.random_quaternion(self._rng.rand(3))
            self._world.add_model(path, position, orientation, scaling=scale)
            self._world.run(0.4)

        # Wait for the objects to rest
        self._world.run(1.)

class WithBody(BaseScene):
    # 3D workspace for tote 1
    def reset(self):
        print("withbody reset")
        self._workspace1_bounds = np.array([
            [0.38, 0.62],  # 3x2 rows: x,y,z cols: min,max
            [-0.22, 0.22],
            [0.00, 0.5]
        ])
        # 3D workspace for tote 2
        self._workspace2_bounds = np.copy(self._workspace1_bounds)
        self._workspace2_bounds[0, :] = - self._workspace2_bounds[0, ::-1]

        # load environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.8)

        # Load objects
        # - possible object colors
        self._object_colors = self.get_tableau_palette()

        # - Define possible object shapes
        self._object_shapes = [
            "assets/objects/cube.urdf",
            "assets/objects/rod.urdf",
            "assets/objects/custom.urdf",
        ]

        self._num_objects = len(self._object_shapes)
        self._object_shape_ids = [
            i % len(self._object_shapes) for i in range(self._num_objects)]
        self._objects_body_ids = []
        for i in range(self._num_objects):
            object_body_id = p.loadURDF(self._object_shapes[i], [ 0.5, 0.1, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
            self._objects_body_ids.append(object_body_id)
            p.changeVisualShape(object_body_id, -1, rgbaColor=[*self._object_colors[i], 1])
        self.reset_objects()

        print("withbody reset out")
            
    def reset_objects(self):
        for object_body_id in self._objects_body_ids:
            random_position = np.random.random_sample((3))*(self._workspace1_bounds[:, 1]-(
                self._workspace1_bounds[:, 0]+0.1))+self._workspace1_bounds[:, 0]+0.1
            random_orientation = np.random.random_sample((3))*2*np.pi-np.pi
            p.resetBasePositionAndOrientation(
                object_body_id, random_position, p.getQuaternionFromEuler(random_orientation))
        #self.step_simulation(2e2)

    def get_tableau_palette(self):
        """
        returns a beautiful color palette
        :return palette (np.array object): np array of rgb colors in range [0, 1]
        """
        palette = np.array(
            [
                [89, 169, 79],  # green
                [156, 117, 95],  # brown
                [237, 201, 72],  # yellow
                [78, 121, 167],  # blue
                [255, 87, 89],  # red
                [242, 142, 43],  # orange
                [176, 122, 161],  # purple
                [255, 157, 167],  # pink
                [118, 183, 178],  # cyan
                [186, 176, 172]  # gray
            ],
            dtype=np.float
        )
        return palette / 255.