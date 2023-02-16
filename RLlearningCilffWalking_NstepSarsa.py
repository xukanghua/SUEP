# 暑期Python学习
# 学生：许康桦
# 练习时间：2022/11/3  9:13
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


class nstepSarsa:
    def __init__(self,n,row,col,epsilon,gamma,alpha,numaction=4):
        self.row=row
        self.col=col
        self.numaction=numaction
        self.Q_table=np.zeros([self.col*self.row,numaction])
        self.alpha=alpha
        self.gamma=gamma
        self.epsilon=epsilon
        #以下是跟sarsa法不同的地方
        self.n=n#表示采用n步sarsa法的n
        self.statelist=[]#保存之前的状态
        self.actionlist=[]#保存之前的动作
        self.rewardlist=[]#保存之前的奖励

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

    def update(self,s0,a0,s1,a1,r,done):#更新Q表与之前的sarsa法很多不一样
        self.actionlist.append(a0)
        self.rewardlist.append(r)
        self.statelist.append(s0)
        if len(self.statelist)==self.n:#n步sarsa法的数据量满足条件则进行n步sarsa法的运算
            G=self.Q_table[s1,a1]
            for i in reversed(range(self.n)):
                G=self.gamma*G+self.rewardlist[i]
                #TDtarget由r_t+λ*Q(s_t+1,a_t+1)变为
                #r_t+λ*r_t+1+.....λ^n*Q(s_t+n,a_t+n)
                #回到了之前计算回报G的思路上，上面的λ跟程序中的衰减因数gamma是一个意思
                if done and i > 0:#若提前到达终点或者掉下悬崖则提前更新Q表
                    s = self.statelist[i]
                    a = self.actionlist[i]
                    self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
            s = self.statelist.pop(0)  # 将需要更新的状态动作从列表中删除,下次不必更新
            a = self.actionlist.pop(0)
            self.rewardlist.pop(0)
            # n步Sarsa的更新步骤
            self.Q_table[s, a] += self.alpha * (G - self.Q_table[s, a])
        if done:#如果到达终止状态则清空列表并开始下一轮
            self.statelist.clear()
            self.actionlist.clear()
            self.rewardlist.clear()


row=4
col=12
env = CliffWalkingEnv(row, col)
np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 0.9
n=5
agent = nstepSarsa(n,row, col, epsilon, gamma, alpha)
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
        agent.update(state, action, next_state, next_action, reward,done)
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
print('5步Sarsa算法最终收敛得到的策略为：')
print_agent(agent, env, action_meaning, list(range(37, 47)), [47],[36])