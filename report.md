### Hyperparameter tuning
All Neural net project has to be adjusted. In this project the following parameters were choosen:
The neural net is a 33x128x128x4 size. (Input 33, 2 hidden layer and the output is 4)

```
BUFFER_SIZE = int(1e5)   # replay buffer size - we can define how many experiences we would like to collect
BATCH_SIZE = 128         # minibatch size - That depends of the memory mainly. The 128 was sufficient for this project. 
GAMMA = 0.99            # discount factor - I left it as a standard
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-3         # learning rate of the actor - We are using Adam optimizer, and with the combination of a buffer size it had a nice speed to converge
LR_CRITIC = 1e-3        # learning rate of the critic - - We are using Adam optimizer, and with the combination of a buffer size it had a nice speed to converge
WEIGHT_DECAY = 0        # L2 weight decay
LR = 1e-3               # learning rate - We are using Adam optimizer, and with the combination of a buffer size it had a nice speed to converge
UPDATE_EVERY = 2        # how often to update the network - I tried with 1-6 and 2 was performing the best

```

### Changes on the algorithm:

I have used gradient clipping when training the critic network as it was suggested in the project description:

```
self.critic_optimizer.zero_grad()
critic_loss.backward()
torch.nn.utils.clip_grad_norm(self.critic_local.parameters(), 1)
self.critic_optimizer.step()
  
```
I have also reduced the number of updates per time step. Instead of updating the actor and critic networks 20 times at every timestep, I amended the code to update the networks 10 times after every 10 timesteps.

### The training result

![result](https://github.com/plitkei/RL-Unity-MLAgents-Python-Pytorch-reacher.app/blob/main/result.jpg)

### Future improvements

Better hyperparameter tuning, using PPO, A3C or D4PG. I think batch normalization would also help. 

