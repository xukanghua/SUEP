# 暑期Python学习
# 学生：许康桦
# 练习时间：2022/11/2  10:43
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


class Sarsa:
    def __init__(self,row,col,epsilon,gamma,alpha,numaction=4):
        self.row=row
        self.col=col
        self.numaction=numaction
        self.Q_table=np.zeros([self.col*self.row,numaction])
        self.alpha=alpha
        self.gamma=gamma
        self.epsilon=epsilon

    def takeaction(self,state):
        if np.random.random()<self.epsilon:
            action=np.random.randint(self.numaction)
        else:
            action=np.argmax(self.Q_table[state])
        return action
    def bestaction(self,state):
        Q_max=np.max(self.Q_table[state])
        a=[0,0,0,0]
        for i in range(self.numaction):
            if self.Q_table[state,i]==Q_max:
                a[i]=1
        return a

    def update(self,s0,a0,s1,a1,r):#更新Q表
        tderror=r+self.gamma*self.Q_table[s1,a1]-self.Q_table[s0,a0]
        self.Q_table[s0,a0]+=self.alpha*tderror


row=4
col=12
env = CliffWalkingEnv(row, col)
np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 0.9
agent = Sarsa(row, col, epsilon, gamma, alpha)
num_episodes = 500  # 训练次数
lst = []
for i_episode in range(num_episodes):
    state = env.reset()
    action = agent.takeaction(state)
    done = False
    rewardsum=0
    while not done:
        next_state, done, reward = env.step(action)
        rewardsum = rewardsum+reward#记录训练过程中的回报
        next_action = agent.takeaction(next_state)
        agent.update(state, action, next_state, next_action, reward)
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
            elif (i*env.col+j) in start:
                print('S',end=' ')
            else:
                a=agent.bestaction(i*env.col+j)
                pi_str=''
                for k in range(len(action_meaning)):
                    if a[k] > 0:
                        pi_str += action_meaning[k]
                print(pi_str,end=' ')
        print()#换行


action_meaning = ['^', 'v', '<', '>']
print('Sarsa算法最终收敛得到的策略为：')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47],[36])

