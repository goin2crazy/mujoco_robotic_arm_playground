from stable_baselines3 import SAC 
import torch 

from observation import get_observation_reach_task

class ReachingAgent(): 
    def __init__(self, model_path):
        
        self.model = SAC.load(model_path)

    def __call__(self, *args, **kwds):
        obs = get_observation_reach_task(*args, **kwds)
        with torch.no_grad(): 
            reaching_direction = self.model.predict(obs, deterministic=True)[0]

        return reaching_direction