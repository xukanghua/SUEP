import random
import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import collections
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'#输出信息ERROR + FATAL


class ReplayBuffer:#回放经验池
    def __init__(self, capacity=10000):#经验池大小
        self.buffer=collections.deque(maxlen=capacity)#使用队列存放经验数据，先进先出

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self,batch_size=32):
        sample=random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, done=map(np.asarray, zip(*sample))#zip(*)表示解包，map会根据提供的函数对指定的序列做映射，np.asarray表示将输入转换为数组
        #states=np.array(states).reshape(batch_size, 4)           #随机提取32个数据用于神经网络训练
        #next_states=np.array(next_states).reshape(batch_size, 4)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)   #返回经验池大小


class ActionStateModel:#动作状态模型
    def __init__(self, state_dim, aciton_dim):
        self.state_dim=state_dim#状态维度
        self.action_dim=aciton_dim#动作维度
        self.epsilon=1.0#初始贪心策略因子

        self.model=self.create_model()

    def create_model(self):#创建神经网络模型，并返回模型
        model=tf.keras.Sequential([#序列化模型来构建神经网络
            tf.keras.layers.Input((self.state_dim,)),#输入层，输入为四个状态
            tf.keras.layers.Dense(32, activation='relu'),#定义全连接层，节点数32，神经元激活函数relu即max(0, x)
            tf.keras.layers.Dense(16, activation='relu'),#同上
            tf.keras.layers.Dense(self.action_dim)#输出层
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=0.005))#配置训练方法，损失函数为均方误差，学习率为0.005
        return model

    def predict(self, state):
        return self.model.predict(state)#通过神经网络得到Q值

    def get_action(self, state):#采取动作
        state=np.reshape(state, [1, self.state_dim])
        self.epsilon*=0.995
        self.epsilon=max(self.epsilon, 0.01)#贪心策略因子最小值为0.01
        q_value=self.predict(state)[0]#二维列表变为一维
        if np.random.random()<self.epsilon:
            return random.randint(0, self.action_dim-1)#任选左右
        return np.argmax(q_value)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0)#迭代次数为1，verbose=0则不输出日志


class Agent:#定义智能体类
    def __init__(self, env):
        self.env=env
        self.state_dim=self.env.observation_space.shape[0]
        self.action_dim=self.env.action_space.n
        self.model=ActionStateModel(self.state_dim, self.action_dim)
        self.target_model=ActionStateModel(self.state_dim, self.action_dim)#创建两个一样的神经网络，一个是每次都更新的训练网络，一个是隔一段时间更新的目标网络
        self.target_update()
        self.buffer=ReplayBuffer()

    def target_update(self):
        weights=self.model.model.get_weights()
        self.target_model.model.set_weights(weights)

    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done=self.buffer.sample()
            targets=self.target_model.predict(states)
            next_q_values=self.target_model.predict(next_states).max(axis=1)
            #a=rewards+(1-done)*next_q_values*0.95
            targets[range(32), actions]=rewards+(1-done)*next_q_values*0.95
            #targets形状为（32，2），与神经网络Q值输出维度一致
            self.model.train(states, targets)



env=gym.make('CartPole-v0')
agent=Agent(env)
episode_list=[]
return_list=[]
for ep in range(200):
    done, total_reward=False, 0
    state=env.reset()
    while not done:
        action=agent.model.get_action(state)
        #env.render()#render表示重绘环境的一帧
        next_state, reward, done, _=env.step(action)
        agent.buffer.put(state, action, reward*0.01, next_state, done)
        total_reward+=reward
        state=next_state
    if agent.buffer.size()>=100:
        agent.replay()
    agent.target_update()
    return_list.append(total_reward)
    episode_list.append(ep)


plt.plot(episode_list,return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on CartPole-v0')
plt.show()






