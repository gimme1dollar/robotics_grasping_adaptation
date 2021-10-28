from enum import Enum
import pybullet as p
import numpy as np
import random
import time
import gym

from gym.utils import seeding
from numpy.random import RandomState
from agent.world.model import Model
from agent.world import scene
from pybullet_utils import bullet_client
#from numba import cuda

class World(gym.Env):
    class Events(Enum):
        RESET = 0
        STEP = 1

    def __init__(self, config, evaluate, test, validate):
        #print("world init")
        """Initialize a new simulated world.

        Args:
            config: A dict containing values for the following keys:
                real_time (bool): Flag whether to run the simulation in real time.
                visualize (bool): Flag whether to open the bundled visualizer.
        """
        # Config
        self._rng = self.seed(evaluate=evaluate)
        config = config['simulation']
        config_scene = config['scene']
        self.scene_type = config_scene['scene_type']
        self.object_num = config_scene['object_num']

        # Pybullet client
        visualize = config.get('visualize', True) 
        self._real_time = config.get('real_time', True)
        self.physics_client = bullet_client.BulletClient(
            p.GUI if visualize else p.DIRECT)

        # Time
        self.epoch = 0
        self.sim_time = 0.
        self._time_step = 1. / 240.
        self._time_horizon = config['time_horizon']
        self._solver_iterations = 150 

        # Scene
        if self.scene_type == "OnFloor":
            self._scene = scene.OnFloor(self, config, self._rng, test, validate)
        elif self.scene_type == "OnTable":
            self._scene = scene.OnTable(self, config, self._rng, test, validate)
        elif self.scene_type == "OnTote":
            self._scene = scene.OnTote(self, config, self._rng, test, validate)

        # Objects (including robot)
        self.models = []
        self.objects = []

        # callbacks
        self._callbacks = {World.Events.RESET: [], World.Events.STEP: []}


    ## Running simulation
    def run(self, duration):
        for _ in range(int(duration / self._time_step)):
            self.step_sim(1)

    def step_sim(self, num_steps):
        """Advance the simulation by one step."""
        for i in range(int(num_steps)):
            p.stepSimulation()
            
        # self._trigger_event(World.Events.STEP)
        self.sim_time += self._time_step
        if self._real_time:
            time.sleep(max(0., self.sim_time - time.time() + self._real_start_time))

    def reset_sim(self):
        # self._trigger_event(World.Events.RESET) # Trigger reset func
        self.physics_client.resetSimulation()
        self.physics_client.setPhysicsEngineParameter(
            fixedTimeStep=self._time_step,
            numSolverIterations=self._solver_iterations,
            enableConeFriction=1)
        
        # set gravity
        self.physics_client.setGravity(0., 0., -9.81)   

        # set time
        self.epoch += 1
        self.sim_time = 0.
        self._real_start_time = time.time()
 
        # models
        self.models = []
        self._scene.reset()

    def close(self):
        self.physics_client.disconnect()


    ## Models
    def add_model(self, path, start_pos, start_orn, scaling=1.):
        model = Model(self.physics_client)
        model.load_model(path, start_pos, start_orn, scaling)
        self.models.append(model)
        return model

    def remove_model(self, model_id):
        self.physics_client.removeBody(model_id)
        self.models[model_id] = False


    ## Objects (task)
    def reset_objects(self):
        # - Define possible object shapes
        object_shapes = self.shape_objects()
        object_colors = self.color_objects()

        # load objects
        self.objects = []
        
        _object_body_id = p.loadURDF(object_shapes[-1], [2.0, 0.1, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
        self.objects.append(_object_body_id) 
        for i in range(self.object_num):
            _object_shape = random.choice(object_shapes[:-1])
            _object_body_id = p.loadURDF(_object_shape, [2.0, 0.1, 0.1], p.getQuaternionFromEuler([0, 0, 0]))
            self.objects.append(_object_body_id)
            
        # set objects configuration
        for object_body_id in self.objects:
            # poisition
            random_position = [0.7, 0.0, 0.1]
            random_orientation = np.random.random_sample((3))*2*np.pi-np.pi
            p.resetBasePositionAndOrientation(object_body_id, random_position, p.getQuaternionFromEuler(random_orientation))

        # set objects colors
        p.changeVisualShape(self.objects[0], -1, rgbaColor=np.concatenate((object_colors[-1], np.array([1.0]))))
        for object_body_id in self.objects[1:]:
            _object_colors = random.choice(object_colors[:-1])
            _object_colors = np.concatenate((_object_colors, np.array([1.0])))
            p.changeVisualShape(object_body_id, -1, rgbaColor=_object_colors)

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
    

    ## Misc.    
    def seed(self, seed=None, evaluate=False, validate=False):
        if evaluate:
            self._validate = validate
            # Create a new RNG to guarantee the exact same sequence of objects
            self._rng = RandomState(1)
        else:
            self._validate = False
            #Random with random seed
            self._rng, seed = seeding.np_random(seed)
        return self._rng
