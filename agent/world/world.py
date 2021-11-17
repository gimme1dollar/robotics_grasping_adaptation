from enum import Enum
import pybullet as p
import numpy as np
import random
import time
import gym

from gym.utils import seeding
from numpy.random import RandomState
from agent.world.model import Model
from agent.world import task
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
        #config = config['simulation']
        config_scene = config['scene']
        self.scene_type = config_scene['scene_type']

        # Pybullet client
        config_sim = config['simulation']
        visualize = config_sim.get('visualize', True) 
        self._real_time = config_sim.get('real_time', True)
        self.physics_client = bullet_client.BulletClient(p.GUI if visualize else p.DIRECT)

        # Time
        self.epoch = 0
        self.sim_time = 0.
        self._time_step = 1. / 240.
        self._time_horizon = config_sim['time_horizon']
        self._shifted_gravity = config_sim['gravity']
        self._shifted_time = config_sim['unit_time']
        self._solver_iterations = 150 

        # Scene
        if self.scene_type == "OnFloor":
            self._scene = task.OnFloor(self, config, self._rng, test, validate)
        elif self.scene_type == "OnTable":
            self._scene = task.OnTable(self, config, self._rng, test, validate)
        elif self.scene_type == "OnTray":
            self._scene = task.OnTray(self, config, self._rng, test, validate)

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
        if self._shifted_time: num_steps *= 3
        for _ in range(int(num_steps)):
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
        gravity = -9.81
        if self._shifted_gravity: gravity *= 1/10
        self.physics_client.setGravity(0., 0., gravity)   

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
