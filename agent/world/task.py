import os
import pybullet as p
import numpy as np
import random
import pybullet_data
import time
from abc import ABC, abstractmethod
from agent.utils import transform_utils

class BaseTask(ABC):
    def __init__(self, world, config, rng, test=False, validate=False):
        self._world = world
        self._rng = rng
        self._model_path = pybullet_data.getDataPath()
        self._validate = validate
        self._test = test
        self._num_objects = config['scene']['object_num']

        object_samplers = {'wooden_blocks': self._sample_wooden_blocks,
                           'random_urdfs': self._sample_random_objects}
        self._object_sampler = object_samplers[config['scene']['data_set']]
        print("dataset", config['scene']['data_set'])

    ## Objects (task)
    def _sample_wooden_blocks(self, n_objects):
        self._model_path = "models/"
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

    def shape_objects(self):
        """
        returns a shape of objects
        :return palette (np.array object): np array of rgb colors in range [0, 1]
        """

        shapes = [
            "assets/objects/rod.urdf",
            "assets/objects/custom.urdf",
            "assets/objects/cuboid0.urdf",
            "assets/objects/cuboid1.urdf",
            "assets/objects/cylinder.urdf",
            "assets/objects/triangle.urdf",

            "assets/objects/cube.urdf",
        ]

        return shapes

    def color_objects(self):
        """
        returns a beautiful color palette
        :return palette (np.array object): np array of rgb colors in range [0, 1]
        """

        palette = np.array(
            [
                #[78, 121, 167],  # blue
                #[89, 169, 79],  # green
                [237, 201, 72],  # yellow
                [156, 117, 95],  # brown
                [242, 142, 43],  # orange
                #[176, 122, 161],  # purple
                [255, 157, 167],  # pink
                #[118, 183, 178],  # cyan
                #[186, 176, 172],  # gray
                
                [255, 87, 89],  # red
                
            ],
            dtype=np.float
        )

        return palette / 255.

    @abstractmethod
    def reset(self):
        raise NotImplementedError


class OnFloor(BaseTask):
    # 3D workspace for tote 1
    def reset(self):
        self._workspace_bounds = np.array([
            [0.38, 0.62],  # 3x2 rows: x,y,z cols: min,max
            [-0.22, 0.22],
            [0.00, 0.5]
        ])

        # load environment
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self._plane_id = p.loadURDF("../assets/plane/plane.urdf")
        p.setGravity(0, 0, -9.8)

class OnTable(BaseTask):
    """Tabletop settings with geometrically different objects."""
    def reset(self):
        self.table_path = '../assets/table/table.urdf'
        self.plane_path = '../assets/plane/plane.urdf'
        self._model_path = pybullet_data.getDataPath()
        plane_urdf = os.path.join("models", self.plane_path)
        table_urdf = os.path.join("models", self.table_path)
        self._world.add_model(plane_urdf, [0., 0., -1.], [0., 0., 0., 1.])
        self._world.add_model(table_urdf, [0., 0., -.82], [0., 0., 0., 1.])
        
        # Sample random objects
        self._num_objects = self._rng.randint(self._num_objects-2, self._num_objects+2)
        object_shapes = self.shape_objects()
        object_colors = self.color_objects()

        # load objects
        self.objects = []
        _object_body_id = p.loadURDF(object_shapes[-1], [2.0, 0.1, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
        self.objects.append(_object_body_id) 
        for i in range(self._num_objects):
            _object_shape = random.choice(object_shapes[:-1])
            _object_body_id = p.loadURDF(_object_shape, [2.0, 0.1, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
            self.objects.append(_object_body_id)
            
        # set objects configuration
        for object_body_id in self.objects:
            # poisition
            random_position = np.r_[self._rng.uniform(-0.15, 0.15, 2), 0.1]
            random_orientation = np.random.random_sample((3))*2*np.pi-np.pi
            p.resetBasePositionAndOrientation(object_body_id, random_position, p.getQuaternionFromEuler(random_orientation))

        # set objects colors
        p.changeVisualShape(self.objects[0], -1, rgbaColor=np.concatenate((object_colors[-1], np.array([1.0]))))
        for object_body_id in self.objects[1:]:
            _object_colors = random.choice(object_colors[:-1])
            _object_colors = np.concatenate((_object_colors, np.array([1.0])))
            p.changeVisualShape(object_body_id, -1, rgbaColor=_object_colors)

        # Wait for the objects to rest
        self._world.run(1.)

class OnTote(BaseTask):
    """Tote settings"""
    def reset(self):
        raise BaseException
        