# 暑期Python学习
# 学生：许康桦
# 练习时间：2022/10/30  19:06


import copy


class CliffWalkingEnv:
    """ 悬崖漫步环境"""
    #总共有48个位置，每个位置都有四个动作以及动作对应的下个位置的状态、奖励和是否完成指示，该部分用元组()构成
    def __int__(self, col=12, row=4):  # row为行，col为列
        self.col = col  # 定义网格世界的列
        self.row = row
        self.P = self.createP()

    def createP(self):
        # 创建转移矩阵P,P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励。第一个p是什么意思？
        P = [[[] for i in range(self.row)] for j in range(self.col * self.row)]
        # 左上角为原点。定义四种动作
        move=[[0,-1],[0,1],[-1,0],[1,0]]  # 分别为向上，向下，向左，向右
        for i in range(self.row):
            for j in range(self.col):
                for a in range(4):
                    if j>0 and i==self.row-1:#在悬崖上的点奖励为0，并且结束
                        P[i*self.col+j][a]=[1,i*self.col+j,0,True]
                        continue
                    next_x=min(self.col-1,max(0,j+move[a][0]))#防止x跑出范围
                    next_y=min(self.row-1,max(0,i+move[a][1]))#防止y跑出范围
                    next_state=next_y*self.col+next_x#将下个状态的行列值转换为0~48之间的数
                    reward=-1
                    done=False
                    if next_y==self.row-1 and next_x>0:
                        done=True
                        if next_x!=self.col-1:
                            reward=-100   #下一步掉下悬崖
                    P[i * self.col + j][a] = [(1, next_state, reward, done)]  #当前位置的下一个状态和奖励
        return P
# class CliffWalkingEnv:
#     """ 悬崖漫步环境"""
#       #总共有48个位置，每个位置都有四个动作以及动作对应的下个位置的状态、奖励和是否完成指示，该部分用元组()构成
#     def __init__(self, col=12, row=4):# row为行，col为列
#         self.col = col  # 定义网格世界的列
#         self.row = row  # 定义网格世界的行
#         # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
#         self.P = self.createP()
#
#     def createP(self):
#         # 初始化
#         P = [[[] for j in range(4)] for i in range(self.row * self.col)]
#         # 左上角为原点。定义四种动作
#         change = [[0, -1], [0, 1], [-1, 0], [1, 0]]# 分别为向上，向下，向左，向右
#         for i in range(self.row):
#             for j in range(self.col):
#                 for a in range(4):
#                     # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
#                     if i == self.row - 1 and j > 0:#在悬崖上的点奖励为0，并且结束
#                         P[i * self.col + j][a] = [(1, i * self.col + j, 0,True)]
#                         continue
#                     # 其他位置
#                     next_x = min(self.col - 1, max(0, j + change[a][0]))#防止x跑出范围
#                     next_y = min(self.row - 1, max(0, i + change[a][1]))#防止y跑出范围
#                     next_state = next_y * self.col + next_x#将下个状态的行列值转换为0~48之间的数
#                     reward = -1
#                     done = False
#                     # 下一个位置在悬崖或者终点
#                     if next_y == self.row - 1 and next_x > 0:
#                         done = True
#                         if next_x != self.col - 1:  # 下一个位置在悬崖
#                             reward = -100 #下一步掉下悬崖
#                     P[i * self.col + j][a] = [(1, next_state, reward, done)]#当前位置的下一个状态和奖励
#         return P

class PolicyIteration:
    """ 策略迭代算法 """
    def __init__(self,env,theta,gamma):#初始化
        self.env=env
        self.v = [0] * self.env.col * self.env.row  # 初始化价值为0
        self.pi=[[0.25,0.25 ,0.25 ,0.25] for i in range(self.env.row*self.env.col)]
        self.theta=theta#策略评估收敛值
        self.gamma=gamma#折扣因子

    def policy_evaluation(self):#策略评估
        cnt=1
        while 1:
            max_diff=0
            new_v=[0]*self.env.row*self.env.col
            for s in range(self.env.row*self.env.col):
                qsa_list=[]
                for a in range(4):
                    qsa=0
                    for res in self.env.P[s][a]:#得到第s个点的4个方向的元组之一
                        p,next_state,r,done=res      #将元组中的数据提取出来
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))#还是不知道p干嘛用
                        #当前状态下的所有动作价值之和
                        #q(s,a)=即时奖励r+s状态下的所有动作求和（s状态做动作a到s'状态转移概率*s'的状态价值（初始均为0，随便设，后面会收敛））
                    qsa_list.append(self.pi[s][a] * qsa)
                    #根据贝尔曼期望方程得到当前状态s，当前策略下该动作的价值=策略*q(s,a)
                new_v[s] = sum(qsa_list)#根据贝尔曼期望方程得到当前状态价值v(s)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))#求v(s)的最大差值
            self.v = new_v#更新状态价值函数
            if max_diff < self.theta: break  # 满足收敛条件,退出评估迭代
            cnt+=1  #策略评估迭代次数
        print("策略迭代一共进行%d轮" % cnt)

    def policy_improvement(self):#策略提升
        for s in range(self.env.row * self.env.col):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
                #以上部分同上策略评估最后得到当前状态的所有动作的价值
            maxq = max(qsa_list)#选取价值最大的动作
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            # 让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]#更新策略，策略只选择最大值的动作，值一样的话就均分概率
        return self.pi  #返回策略

    def policy_iteration(self):  # 策略迭代主程序
        while 1:
            self.policy_evaluation()
            old_pi = copy.deepcopy(self.pi)  # 将列表进行深拷贝,方便接下来进行比较
            #深拷贝：深拷贝不会拷贝引用类型的引用，而是将引用类型的值全部拷贝一份，形成一个新的引用类型，
            # 这样就不会发生引用错乱的问题，使得我们可以多次使用同样的数据，而不用担心数据之间会起冲突。
            new_pi = self.policy_improvement()
            if old_pi == new_pi: break#策略不再提升则终止

def print_agent(agent, action_meaning, disaster=[], end=[]):#用于显示结果
    print("状态价值：")
    for i in range(agent.env.row):
        for j in range(agent.env.col):
            # 为了输出美观,保持输出6个字符
            print('%6.6s' % ('%.3f' % agent.v[i * agent.env.col + j]), end=' ')
        print()

    print("策略：")
    for i in range(agent.env.row):
        for j in range(agent.env.col):
            # 一些特殊的状态,例如悬崖漫步中的悬崖
            if (i * agent.env.col + j) in disaster:
                print('****', end=' ')
            elif (i * agent.env.col + j) in end:  # 目标状态
                print('EEEE', end=' ')
            else:
                a = agent.pi[i * agent.env.col + j]
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()

env = CliffWalkingEnv()
action_meaning = ['^', 'v', '<', '>']
theta = 0.001
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])