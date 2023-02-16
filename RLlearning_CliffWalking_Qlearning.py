# 暑期Python学习
# 学生：许康桦
# 练习时间：2022/11/1  18:35
import numpy as np
import matplotlib.pyplot as plt

class CliffWalkingEnv:
    def __init__(self,row,col):
        self.row=row
        self.col=col
        self.x=0
        self.y=self.row-1
    def step(self,action):#没进行一个动作返回下一个状态、奖励和完成态
        change=[[0,-1],[0,1],[-1,0],[1,0]]#顺序为上下左右
        self.x=min(self.col-1,max(0,self.x+change[action][0]))
        self.y=min(self.row-1,max(0,self.y+change[action][1]))
        reward=-1
        next_state=self.y*self.col+self.x
        done=False
        if self.x>0 and self.y==self.row-1:
            done=True
            if self.x!=self.col-1:
                reward=-100
        return next_state,done,reward

    def reset(self):
        self.x=0
        self.y=self.row-1
        return self.x+self.y*self.col

class QLearning:
    """ Q-learning算法 """
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数

    def takeaction(self, state):  #选取下一步的操作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def bestaction(self, state):  # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1):#Qlearning的Q表更新策略
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

col=12
row=4
env=CliffWalkingEnv(row,col)
np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 0.9
agent=QLearning(col,row,epsilon,alpha,gamma)
num_episodes=500
lst=[]

for i_episode in range(num_episodes):
    state = env.reset()
    action = agent.takeaction(state)
    done = False
    rewardsum=0
    while not done:
        next_state, done, reward = env.step(action)
        rewardsum = rewardsum+reward#记录训练过程中的回报
        next_action = agent.takeaction(next_state)
        agent.update(state, action, reward, next_state)
        state = next_state
        action = next_action
    lst.append(rewardsum)
plt.plot([i for i in range(num_episodes)],lst)
plt.show()



def print_agent(agent,env,action_meaning,disaster=[],end=[],start=[]):
    for i in range(env.row):
        for j in range(env.col):
            if (i*env.col+j) in disaster:
                print('*',end=' ')
            elif (i*env.col+j) in end:
                print('E', end=' ')
            elif (i * env.col + j) in start:
                print('S', end=' ')
            else:
                a=agent.bestaction(i*env.col+j)
                pi_str=''
                for k in range(len(action_meaning)):
                    if a[k] > 0:
                        pi_str += action_meaning[k]
                print(pi_str,end=' ')
        print()#换行


action_meaning = ['^', 'v', '<', '>']
print('Qlearning算法最终收敛得到的策略为：')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47],[36])