# adaptation-on-grasping
Domain adaptation on robot grasping 


## Environment
- environment
```
- Windows10
- nvidia-dirver 461.72
- cuda 11.1
- cuDNN 8.0.5
- python 3.8
- pybullet 2.6.4
- tensorflow 1.14.0
- keras 2.2.4
- stable_baseline 2.10.2
```
- installation 
```
git clone <url_repo> <dir_name>
conda create -n <env_name> 
conda activate <env_name> 
pip install -r requirements.txt
```

- robot configuration (references)       
https://github.com/BarisYazici/deep-rl-grasping     

```
pytest test
```

## Algorithm
- Pseudo algorithm for reinforcement learning
<p align="center">
<img src="demo/sac_pseudo.png" width="400px" height="215px">
</p>

- Domain adaptation pseudo algorithm
<p align="center">
(Currently Working)
</p>

## Demo 
```
python main.py run --config config/<conf_name> --model checkpoints/final/<path_name>
```

- simulation
<p align="center">
<img src="demo/gripper_demo.gif" width="350px">
</p>  
<p align="center">
<img src="demo/arm_demo.gif" width="350px">
</p>  

- baseline training
<p align="center">
<img src="demo/gripper_learning_curve.PNG" width="500px" height="250px">
</p>  

- domain shift
<p align="center">
<img src="demo/domain_shift.PNG" width="320px" height="100px">
</p>  
<p align="center">
<img src="demo/domain_shift.gif" width="500px" height="300px">
</p>  


## Contributors
이주용 안석준