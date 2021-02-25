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

### DDPG ALgorithm - Deep Deterministic Policy gradient

The DDPG is a policy based method and not a value-based as the DQN was in the last project. The policy-based methods directly learn the optimal policy, without having to maintain a separate value function estimate as we did at last time. So we had an in between step to obtain the optimal policy from the optimal action-value function.

#### OK. And? Why are we not using here also the DQN based model?

There are several reason for that:
* The main one is that the DQN produces a discrete action selection. We could try to solve the environment by discretizing the continous output, but it would be too difficult to learn. The policy based methods are well suited for continous action spaces. (Continous vs discrete: we do not only have a choice of move left or right, but in more like "how much left or right" where the how much is in a continous space like between -100 and +100)
* It is simpler because we are not need to store additional data, like action values
* It can learn true stochastic policies

#### Learning the Optimal policy directly OK. But what is "Gradient"?

Policy gradient methods are a subclass of policy-based methods that estimate the weights of an optimal policy through gradient ascent.
Yes, ascent and not descent as we have used in all deep learning projects. The ascent means that we are not minimizing a loss function, but maximaizing the expected return.

#### D for Deep, P - Policy, G - gradient.

What is the extra D at the begining - Deterministic. It means that for a certain state we will get always the same action. If it would be Stochastic Policy - we would get a probability for a state.

#### Hmm. But we have used 2 networks one of them was Actor another one was Critic..

Well, actually we have used 2 actor and 2 Critic networks, but first what is Actor - Critic in a nutshell:

Actor - Critic is marrying the the two world of Policy and Value methods. 
The policy based methods reinforce the good actions an penalize the bad ones. After a lots of experience we have increased the probabilty of actions that led to a win. And decreased the ones that led to losses. (problem: High varience, slow learner)
The value based are always guessing - like the last project DQN compared a guess with a guess... (problem: Introduces bias and over or under estimation)

If you would dig more into the Actor-Critic methods, you might say that DDPG is not really one.

At DDPG what is the role of the two types of networks?
Actor: Optimal policy, deterministicly - outputs the best believed action for any given state 
Critic: Learns evaluates the optimal action - value function  

In the DQN we had a local and the target network. And after a bunch of time steps we just copied the parameters from the local to the target network. 
In DDPG we also have to maintain the local and the target Critic networks, and from the local to target copy the parameters. But we not copying in big chunks, instead of we are using a soft update: We "blending" a very small portions from the local network to the target at each step. (Like 0.01%)
Same for Actor and Critic networks.

We get faster convergence using this startegy. The soft update can actually used at DQN as well.

More details in the code, with comments.

### The training result

![result](https://github.com/plitkei/RL-Unity-MLAgents-Python-Pytorch-reacher.app/blob/main/result.jpg)

### Future improvements

Better hyperparameter tuning, using PPO, A3C or D4PG. I think batch normalization would also help. 

