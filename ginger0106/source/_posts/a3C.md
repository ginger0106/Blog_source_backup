
# A3C  (asynchronous advantage actor-critic)

>Mnih V, Badia A P, Mirza M, et al. Asynchronous methods for deep reinforcement learning[C]//International Conference on Machine Learning. 2016: 1928-1937.
2016 ICML

# 1. on-policy& off-policy
Sutton的书RL an introduction edition 2. 在5.4节 Monte Carlo Control without Exploring starts中，作者定义了on-policy与off-policy:

>On-policy methods attempt to evaluate or improve the policy that is used to make decisions, 
whereas off-policy methods evaluate or improve a policy different from that used to generate the data.

一句话解释：其实就是只有一句话: 更新值函数时是否只使用当前策略所产生的样本. 

![](微信截图_20180327205214.png)
- on-policy：REINFORCE, TRPO, SARSA

- off-policy：Q-learning, Deterministic policy gradient

![](http://jiantuku-image-ginger.oss-cn-beijing.aliyuncs.com/18-6-9/35231674.jpg)
![](http://jiantuku-image-ginger.oss-cn-beijing.aliyuncs.com/18-6-9/96174817.jpg)

# 2. Value-based RL & Policy-based RL 


## Def

- **Value-based： ** A policy is generated from value function:
$$ V_\theta (s)= V^\pi(s) $$
$$ Q_\theta (s,a)= Q^\pi(s,a)$$

> 我们用线性非线性的方式对值函数进行逼近求解

- **Policy-based RL： **Directly parametrise the policy: 
$$ \pi_\theta(s,a)= P[a | s,\theta]$$
 The output is a probability. 
 
>用线性或非线性（如神经网络）对策略进行求解 
 
## Difference 
- Value Based 
  - Learnt Value Function
  - Implicit policy
  
  > 先求值函数，然后用贪婪算法求策略，所以是隐含的：

- Policy Based
  - No Value Function
  - Learnt Policy
  
  
- Actor-Critic
  - Learnt Value Function
  - Learnt Policy

![](http://jiantuku-image-ginger.oss-cn-beijing.aliyuncs.com/18-6-9/31700400.jpg)

##  Policy-Based RL

- Advantages:
  - Better convergence properties
  >基于策略的学习可能会具有更好的收敛性，这是因为基于策略的学习虽然每次只改善一点点，但总是朝着好的方向在改善；而价值函数在后期可能会一直围绕最优价值函数持续小的震荡而不收敛，同时有时候求解值函数非常复杂。
  - Effective in high-dimensional or continuous action spaces
  - Can learn stochastic policies
- Disadvantages:
  - Typically converge to a local rather than global optimum
  - Evaluating a policy is typically inefficient and high variance
  >原始基于策略的学习效率不够高，还有较高的变异性（方差，Variance）。因为基于价值函数的策略决定每次都是推促个体去选择一个最大价值的行为；但是基于策略的，更多的时候策略的选择时仅会在策略某一参数梯度上移动一点点，使得整个的学习比较平滑，因此不够高效。同时当评估单个策略并不充分时，方差较大。

  
## Policy Optimisation
- Policy based reinforcement learning is an optimisation problem

所以这是一个优化问题，我们要做的是利用参数化的策略函数，通过调整这些参数来得到一个较优策略，遵循这个策略产生的行为将得到较多的奖励。具体的机制是设计一个目标函数，对其使用梯度上升（Gradient Ascent）算法优化参数以最大化奖励。

- Find θ that maximises J(θ)

而将策略表达成参数θ的目标函数，有如下几种形式，start value是针对拥有起始状态的情况下求起始状态 $s_1$ 获得的奖励，average value针对不存在起始状态而且停止状态也不固定的情况，在这些可能的状态上计算平均获得的奖励。 Average reward per time-step为每一个时间步长在各种情况下所能得到的平均奖励。
> 目标函数的本质还是奖励。



- Compute Policy Gradient

如何来优化这个目标？ 我们采用随即梯度算法来解决。

![](http://jiantuku-image-ginger.oss-cn-beijing.aliyuncs.com/18-6-9/16343231.jpg)
![](http://jiantuku-image-ginger.oss-cn-beijing.aliyuncs.com/18-6-9/58209890.jpg)
>所以，现在我们得到所有形式的目标函数所对应的策略梯度是一样的，注意这里有两个部分组成，一个是策略函数的log形式，一个是引导奖励(score function)。第一部分是参数化直接得到的，第二部分可以直接用即时奖励来计算，也可以用值函数近似，也就是AC算法。

## Policy Function
- softmax
- Gaussian 
- Random 

## Score Function
根据Score Function的不同，对应不同的算法。如下图：

![](http://jiantuku-image-ginger.oss-cn-beijing.aliyuncs.com/18-6-9/88031244.jpg)

- Monte-Carlo Policy Gradient (REINFORCE)
Using return vt as an unbiased sample of $Q^{\pi_\theta}(s_t, a_t)$ 
> Monte-Carlo policy gradient still has high variance


# 3. Actor-Critic  

## Q Actor-Critic 

- We use a critic to estimate the action-value function
$$ Q_w(s,a)\approx Q^{\pi_\theta}(s, a)$$

- Actor-critic algorithms maintain two sets of parameters

  -**Critic** Updates action-value function parameters w
  
  -**Actor** Updates policy parameters θ, in direction suggested by critic
  
 $$\Delta\theta=\alpha\nabla_\theta log \pi_\theta(s,a)Q_w(s, a)$$
 
 >可以明显看出，Critic做的事情其实是我们已经见过的：策略评估，他要告诉个体，在由参数 $\theta$ 确定的策略 $\pi_{\theta}$ 到底表现得怎么样。关于策略评估我们之前讲过，可以使用蒙特卡洛策略评估、TD学习以及TD(λ)等方式实现
 
![AC](http://jiantuku-image-ginger.oss-cn-beijing.aliyuncs.com/18-6-9/45370982.jpg)
 
> 在这个例子当中，我们只是用了线性的函数来估计Q，当然目前都是神经网络来计算了。另外，REINFORCE和Q AC有high variance的问题。因此有了如下的改进算法。

## Advantage Actor-Critic 
- advantage function
$$ V_v(s,a)\approx V^{\pi_\theta}(s, a)$$ 
$$ Q_w(s,a)\approx Q^{\pi_\theta}(s, a)$$ 
$$ A(s,a)= Q_w(s,a)-V_v(s,a)$$
![](微信截图_20180327224706.png)

##  Training
![](http://jiantuku-image-ginger.oss-cn-beijing.aliyuncs.com/18-6-9/10566688.jpg)

如上图：我们说AC结合了 Policy Gradient (Actor) 和 Function Approximation (Critic) 的方法. Actor 基于概率选行为, Critic 基于 Actor 的行为评判行为的得分, Actor 根据 Critic 的评分修改选行为的概率。

结合到具体的网络上就是：Actor和Critic各为一个网络，Actor输入是状态输出的是动作，loss就是$log\_prob * td\_error$,(和策略梯度相对应，注意到这里的loss和Policy Gradient中的差不多，只是vt换成td_error，引导奖励值vt换了来源（Critic给的）而已)，Critic输入的是状态输出的是Q值，loss是$MSE((r+\gamma*Q_{next}) - Q_{eval})$也就是$MSE(td\_error)$，也就是说这里更新critic对应Q-learning是一样的均方误差。



# 4. A3C  

## Asynchronous

DQN比传统RL算法有了巨大提升其中一个主要原因就是使用了经验回放的技巧。然而，打破数据的相关性，经验回放并非是唯一的方法。另外一种方法是**异步**的方法（异步的方法是指数据并非同时产生）
![](微信截图_20180328102925.png)
## Advantage
相比DQN算法，A3C算法不需要使用经验池来存储历史样本并随机抽取训练来打乱数据相关性，节约了存储空间，并且采用异步训练，大大加倍了数据的采样速度，也因此提升了训练速度。与此同时，采用多个不同训练环境采集样本，样本的分布更加均匀，更有利于神经网络的训练。

![](http://jiantuku-image-ginger.oss-cn-beijing.aliyuncs.com/18-6-9/42121203.jpg)


# 5. Example 
## CartPole
CartPole的玩法如下动图所示，目标就是保持一根杆一直竖直朝上，杆由于重力原因会一直倾斜，当杆倾斜到一定程度就会倒下，此时需要朝左或者右移动杆保证它不会倒下来。我们执行一个动作，动作取值为0或1，代表向左或向右移动，返回的observation是一个四维向量，reward值一直是1，当杆倒下时done的取值为False，其他为True，info是调试信息打印为空具体使用暂时不清楚。如果杆竖直向上的时间越长，得到reward的次数就越多。
![](cp_1.gif)


```python
# OpenGym CartPole-v0 with A3C on GPU
# -----------------------------------
#
# A3C implementation with GPU optimizer threads.
#
# Made as part of blog series Let's make an A3C, available at
# https://jaromiru.com/2017/02/16/lets-make-an-a3c-theory/
#
# author: Jaromir Janisch, 2017

import numpy as np
import tensorflow as tf

import gym, time, random, threading

from keras.models import *
from keras.layers import *
from keras import backend as K

# -- constants
ENV = 'CartPole-v0'

RUN_TIME = 30
THREADS = 8
OPTIMIZERS = 2
THREAD_DELAY = 0.001

GAMMA = 0.99

N_STEP_RETURN = 8
GAMMA_N = GAMMA ** N_STEP_RETURN

EPS_START = 0.4
EPS_STOP = .15
EPS_STEPS = 75000

MIN_BATCH = 32
LEARNING_RATE = 5e-3

LOSS_V = .5  # v loss coefficient
LOSS_ENTROPY = .01  # entropy coefficient


# ---------
class Brain:
    train_queue = [[], [], [], [], []]  # s, a, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self):
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        self.model = self._build_model()
        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())
        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()  # avoid modifications

    def _build_model(self):

        l_input = Input(batch_shape=(None, NUM_STATE))
        l_dense = Dense(16, activation='relu')(l_input)

        out_actions = Dense(NUM_ACTIONS, activation='softmax')(l_dense)
        out_value = Dense(1, activation='linear')(l_dense)

        model = Model(inputs=[l_input], outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, NUM_STATE))
        a_t = tf.placeholder(tf.float32, shape=(None, NUM_ACTIONS))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        p, v = model(s_t)

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = LOSS_V * tf.square(advantage)  # minimize value error
        entropy = LOSS_ENTROPY * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1,
                                               keep_dims=True)  # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < MIN_BATCH:
            time.sleep(0)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < MIN_BATCH:  # more thread could have passed without lock
                return  # we can't yield inside lock

            s, a, r, s_, s_mask = self.train_queue
            self.train_queue = [[], [], [], [], []]

        s = np.vstack(s)
        a = np.vstack(a)
        r = np.vstack(r)
        s_ = np.vstack(s_)
        s_mask = np.vstack(s_mask)

        if len(s) > 5 * MIN_BATCH: print("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + GAMMA_N * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def train_push(self, s, a, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(r)

            if s_ is None:
                self.train_queue[3].append(NONE_STATE)
                self.train_queue[4].append(0.)
            else:
                self.train_queue[3].append(s_)
                self.train_queue[4].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v


# ---------
frames = 0


class Agent:
    def __init__(self, eps_start, eps_end, eps_steps):
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps

        self.memory = []  # used for n_step return
        self.R = 0.

    def getEpsilon(self):
        if (frames >= self.eps_steps):
            return self.eps_end
        else:
            return self.eps_start + frames * (self.eps_end - self.eps_start) / self.eps_steps  # linearly interpolate

    def act(self, s):
        eps = self.getEpsilon()
        global frames;
        frames = frames + 1

        if random.random() < eps:
            return random.randint(0, NUM_ACTIONS - 1)

        else:
            s = np.array([s])
            p = brain.predict_p(s)[0]

            # a = np.argmax(p)
            a = np.random.choice(NUM_ACTIONS, p=p)

            return a

    def train(self, s, a, r, s_):
        def get_sample(memory, n):
            s, a, _, _ = memory[0]
            _, _, _, s_ = memory[n - 1]

            return s, a, self.R, s_

        a_cats = np.zeros(NUM_ACTIONS)  # turn action into one-hot representation
        a_cats[a] = 1

        self.memory.append((s, a_cats, r, s_))

        self.R = (self.R + r * GAMMA_N) / GAMMA

        if s_ is None:
            while len(self.memory) > 0:
                n = len(self.memory)
                s, a, r, s_ = get_sample(self.memory, n)
                brain.train_push(s, a, r, s_)

                self.R = (self.R - self.memory[0][2]) / GAMMA
                self.memory.pop(0)

            self.R = 0

        if len(self.memory) >= N_STEP_RETURN:
            s, a, r, s_ = get_sample(self.memory, N_STEP_RETURN)
            brain.train_push(s, a, r, s_)

            self.R = self.R - self.memory[0][2]
            self.memory.pop(0)

            # possible edge case - if an episode ends in <N steps, the computation is incorrect


# ---------
class Environment(threading.Thread):
    stop_signal = False

    def __init__(self, render=False, eps_start=EPS_START, eps_end=EPS_STOP, eps_steps=EPS_STEPS):
        threading.Thread.__init__(self)

        self.render = render
        self.env = gym.make(ENV)
        self.agent = Agent(eps_start, eps_end, eps_steps)

    def runEpisode(self):
        s = self.env.reset()

        R = 0
        while True:
            time.sleep(THREAD_DELAY)  # yield

            if self.render: self.env.render()

            a = self.agent.act(s)
            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None

            self.agent.train(s, a, r, s_)

            s = s_
            R += r

            if done or self.stop_signal:
                break

        print("Total R:", R)

    def run(self):
        while not self.stop_signal:
            self.runEpisode()

    def stop(self):
        self.stop_signal = True


# ---------
class Optimizer(threading.Thread):
    stop_signal = False

    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while not self.stop_signal:
            brain.optimize()

    def stop(self):
        self.stop_signal = True


# -- main
env_test = Environment(render=True, eps_start=0., eps_end=0.)
NUM_STATE = env_test.env.observation_space.shape[0]
NUM_ACTIONS = env_test.env.action_space.n
NONE_STATE = np.zeros(NUM_STATE)

brain = Brain()  # brain is global in A3C

envs = [Environment() for i in range(THREADS)]
opts = [Optimizer() for i in range(OPTIMIZERS)]

for o in opts:
    o.start()

for e in envs:
    e.start()

time.sleep(RUN_TIME)

for e in envs:
    e.stop()
for e in envs:
    e.join()

for o in opts:
    o.stop()
for o in opts:
    o.join()

print("Training finished")
env_test.run()


```

# Ref

1. <https://zhuanlan.zhihu.com/p/32438022>
2. David Silver ppt
3. https://www.youtube.com/watch?v=O79Ic8XBzvw 
4. https://zhuanlan.zhihu.com/p/32596470
5. http://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
6. http://karpathy.github.io/2016/05/31/rl/



