# 暑期Python学习
# 学生：许康桦
# 练习时间：2022/10/31  21:50
import copy


class CliffWalkingEnv:
    """ 悬崖漫步环境"""
      #总共有48个位置，每个位置都有四个动作以及动作对应的下个位置的状态、奖励和是否完成指示，该部分用元组()构成
    def __init__(self, col=12, row=4):# row为行，col为列
        self.col = col  # 定义网格世界的列
        self.row = row  # 定义网格世界的行
        # 转移矩阵P[state][action] = [(p, next_state, reward, done)]包含下一个状态和奖励
        self.P = self.createP()

    def createP(self):
        # 初始化
        P = [[[] for j in range(4)] for i in range(self.row * self.col)]
        # 左上角为原点。定义四种动作
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]# 分别为向上，向下，向左，向右
        for i in range(self.row):
            for j in range(self.col):
                for a in range(4):
                    # 位置在悬崖或者目标状态,因为无法继续交互,任何动作奖励都为0
                    if i == self.row - 1 and j > 0:#在悬崖上的点奖励为0，并且结束
                        P[i * self.col + j][a] = [(1, i * self.col + j, 0,True)]
                        continue
                    # 其他位置
                    next_x = min(self.col - 1, max(0, j + change[a][0]))#防止x跑出范围
                    next_y = min(self.row - 1, max(0, i + change[a][1]))#防止y跑出范围
                    next_state = next_y * self.col + next_x#将下个状态的行列值转换为0~48之间的数
                    reward = -1
                    done = False
                    # 下一个位置在悬崖或者终点
                    if next_y == self.row - 1 and next_x > 0:
                        done = True
                        if next_x != self.col - 1:  # 下一个位置在悬崖
                            reward = -100 #下一步掉下悬崖
                    P[i * self.col + j][a] = [(1, next_state, reward, done)]#当前位置的下一个状态和奖励
        return P

class ValueIteration:
    """ 价值迭代算法 """
    def __init__(self, env, theta, gamma):
        self.env = env
        self.v = [0] * self.env.col * self.env.row  # 初始化价值为0
        self.theta = theta  # 价值收敛阈值
        self.gamma = gamma
        # 价值迭代结束后得到的策略
        self.pi = [None for i in range(self.env.col * self.env.row)]

    def value_iteration(self):
        cnt = 0
        while 1:
            max_diff = 0
            new_v = [0] * self.env.col * self.env.row
            for s in range(self.env.col * self.env.row):
                qsa_list = []  # 开始计算状态s下的所有Q(s,a)价值
                for a in range(4):
                    qsa = 0
                    for res in self.env.P[s][a]:
                        p, next_state, r, done = res
                        qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                    qsa_list.append(qsa)  # 这一行和下一行代码是价值迭代和策略迭代的主要区别
                new_v[s] = max(qsa_list)
                max_diff = max(max_diff, abs(new_v[s] - self.v[s]))
            self.v = new_v
            if max_diff < self.theta: break  # 满足收敛条件,退出评估迭代
            cnt += 1
        print("价值迭代一共进行%d轮" % cnt)
        self.get_policy()

    def get_policy(self):  # 根据价值函数导出一个贪婪策略
        for s in range(self.env.row * self.env.col):
            qsa_list = []
            for a in range(4):
                qsa = 0
                for res in self.env.P[s][a]:
                    p, next_state, r, done = res
                    qsa += p * (r + self.gamma * self.v[next_state] * (1 - done))
                qsa_list.append(qsa)
            maxq = max(qsa_list)
            cntq = qsa_list.count(maxq)  # 计算有几个动作得到了最大的Q值
            # 让这些动作均分概率
            self.pi[s] = [1 / cntq if q == maxq else 0 for q in qsa_list]

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
agent = ValueIteration(env, theta, gamma)
agent.value_iteration()
print_agent(agent, action_meaning, list(range(37, 47)), [47])