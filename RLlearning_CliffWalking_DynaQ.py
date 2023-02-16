# 暑期Python学习
# 学生：许康桦
# 练习时间：2022/10/31  22:21
import matplotlib.pyplot as plt
import numpy as np
import random



class CliffWalkingEnv:
    """ 悬崖漫步环境"""
  #总共有48个位置，每个位置都有四个动作以及动作对应的下个位置的状态、奖励和是否完成指示，该部分用元组()构成
    def __init__(self, col=12, row=4):# row为行，col为列
        self.col = col  # 定义网格世界的列
        self.row = row  # 定义网格世界的行
        self.x = 0  # 记录当前智能体位置的横坐标
        self.y = self.row - 1  # 记录当前智能体位置的纵坐标

    def step(self,action):
        # 左上角为原点。定义四种动作
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]# 分别为向上，向下，向左，向右
        self.x=min(self.col-1,max(0,self.x+change[action][0]))
        self.y=min(self.row-1,max(0,self.y+change[action][1]))
        next_state=self.x+self.y*self.col
        reward=-1
        done=False
        if self.x>0 and self.y==self.row-1:
            done=True
            if self.x!=self.col-1:
                reward=-100
        return next_state,reward,done

    def reset(self):
        self.x=0
        self.y=self.row-1
        return self.y*self.col+self.x


class DynaQ:
    """ Dyna-Q算法 """
    def __init__(self,col,row,epsilon,alpha,gamma,n_planning,n_action=4):
        self.Q_table = np.zeros([row * col, n_action])  # 初始化Q(s,a)表格
        self.n_action = n_action  # 动作个数
        self.alpha = alpha  # 学习率
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略中的参数
        self.n_planning = n_planning  #执行Q-planning的次数, 对应1次Q-learning
        self.model = dict()  # 环境模型

    def take_action(self, state):  # 选取下一步的操作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def q_learning(self, s0, a0, r, s1):
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error

    def update(self, s0, a0, r, s1):
        self.q_learning(s0, a0, r, s1)
        self.model[(s0, a0)] = r, s1  # 将数据添加到模型中
        for _ in range(self.n_planning):  # Q-planning循环
            # 随机选择曾经遇到过的状态动作对
            (s, a), (r, s_) = random.choice(list(self.model.items()))
            self.q_learning(s, a, r, s_)


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
                a=agent.Q_table[i*env.col+j]
                maxa=max(a)
                pi_str=''
                for k in range(len(action_meaning)):
                    if a[k] == maxa:
                        pi_str += action_meaning[k]
                print(pi_str,end=' ')
        print()#换行



def DynaQ_CliffWalking(n_planning):
    col = 12
    row = 4
    env = CliffWalkingEnv(col, row)
    epsilon = 0.01
    alpha = 0.1
    gamma = 0.9
    agent = DynaQ(col, row, epsilon, alpha, gamma, n_planning)
    num_episodes = 300  # 智能体在环境中运行多少条序列
    lst = []  # 记录每一条序列的回报
    for i_episode in range(num_episodes ):
        episode_return = 0
        state = env.reset()
        done = False
        while not done:
            action = agent.take_action(state)
            next_state, reward, done = env.step(action)
            episode_return += reward  # 这里回报的计算不进行折扣因子衰减
            agent.update(state, action, reward, next_state)
            state = next_state
        lst.append(episode_return)
    print(lst)
    action_meaning = ['^', 'v', '<', '>']
    print('DynaQ算法最终收敛得到的策略为：')
    print_agent(agent, env, action_meaning, list(range(37, 47)), [47], [36])
    plt.plot([i for i in range(num_episodes)], lst)
    plt.show()

return_list = DynaQ_CliffWalking(2)#2步Qplanning


