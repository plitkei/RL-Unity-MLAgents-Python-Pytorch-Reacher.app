# RL-Unity-MLAgents-Python-Pytorch-Reacher.app

### The Environment

For this project, we will work with the Reacher environment.
[image2]: https://user-images.githubusercontent.com/10624937/42386929-76f671f0-8106-11e8-9376-f17da2ae852e.png "Kernel"
[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
![Trained Agent][image1]


In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

We have 20 identical agents, each with its own copy of the environment.

We consider the environment to be solved if the agents get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
* This yields an average score for each episode (where the average is over all 20 agents).

As an example, consider the plot below, where we have plotted the average score (over all 20 agents) obtained with each episode.

### Real-World Robotics

Watch this [YouTube video](https://www.youtube.com/watch?v=ZVIxt2rt1_4) to see how some researchers were able to train a similar task on a real robot! The accompanying research paper can be found [here](https://arxiv.org/pdf/1803.07067.pdf).

### Sharing Experience

In the second version of the project environment, there are 20 identical copies of the agent. It has been shown that having multiple copies of the same agent sharing experience can accelerate learning, and you'll discover this for yourself when solving the project!

### Setting up the environment

I would recommend to choose a fresh environmnet from the current ML-Agent github repo. 
It is a bit more work at the begining, since you have to compile it for your system and you also need to install Unity. 
The problem with the very old versions like this. It is almost immpossible to collect the right (old) versions of the python/pytorch/ML-Agents circle -just to name a few.

If you have the curtesy to use a newer app. In the navigation.ipynb 
do not use the:  *from unityagents import UnityEnvironment*
Use the **from mlagents_envs.environment import UnityEnvironment**
The unityagents are in a different folder in the newer version. So you **Do not** need to pip install unityagents. there is package called like this and that is a very old version of the ML-Agents.

This is how u can set up a new environment:
https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md


*Basically, there is a Unity application what we will call from python with the ML-Agents help. That will be the connection between the unity app and the pytorch code in order to train the Agent.*


We have our Actor - Critic DDPG model in the **model.py** file.
And the DDPG algorithm that uses the model in the **ddpg_agent.py** file.
The connection between unity and the model is in the **Continous_Control.ipynb**

I have used these version of the reacher.app

 **Download the right version:**
 
 * Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
 * Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
 * Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
 * Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    

Place the file in this GitHub repository and unzip (or decompress) the file. Then rename the folder to `Reacher`. (Now `Reacher/Reacher.exe` should exist.)

### Dependencies

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```
	
2. You can clone the Udacity repository, and navigate to the `python/` folder.  Then, install several dependencies.
```bash
git clone https://github.com/udacity/deep-reinforcement-learning.git
cd deep-reinforcement-learning/python
pip install .
```

3. Create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.  
```bash
python -m ipykernel install --user --name drlnd --display-name "drlnd"
```

4. Before running code in a notebook, change the kernel to match the `drlnd` environment by using the drop-down `Kernel` menu. 

![Kernel][image2]


### We are going to use DDPG - Deep Deterministic Policy Gradient to solve this task.

In the **ddpq_agent.py** there is a standard implementation of a DDPG algorithm using two neural net (actor, critic). 
The **model.py** is a "standard" MLP representing our two neural nets. These two files are used in the **Continous_Control.ipynb**

The beauty of these algorithms that you can apply it for other games/problems as well. You just have to adjust the MLP input and output size to adapt to your project.

Normally the weights can be quite big. I decided to upload it because it has a relative small size. Use it if you want to. **checkpoint.pth**

