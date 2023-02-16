import matplotlib.pyplot as plt
import  numpy as np
import gym

class QLearning:
    def __init__(self,ncol,nrow,epsilon,alpha,gamma,n_action=4):
        self.Q_tabel=np.zeros([nrow*ncol,n_action])
        self.n_action=n_action
        self.alpha=alpha #学习率
        self.gamma=gamma #折扣因子
        self.epsilon=epsilon # epsilon-贪婪策略中的参数

    def take_action(self,state):
        if np.random.random()<self.epsilon:
            action=np.random.randint(self.n_action)#随机生成0-3的整数
        else:
            action=np.argmax(self.Q_tabel[state])
        return action


    def update(self,s0,a0,r,s1):
        td_error=r+self.gamma*self.Q_tabel[s1].max()-self.Q_tabel[s0][a0]
        self.Q_tabel[s0,a0]+=self.alpha*td_error


ncol=12
nrow=4
env=gym.make("CliffWalking-v0")

np.random.seed(0)
epsilon = 0.1
alpha = 0.1
gamma = 0.9
agent=QLearning(ncol,nrow,epsilon,alpha,gamma)
num_episodes=500

return_list=[]
for i_episode in range(num_episodes):
    episode_return=0
    state=env.reset()
    done=False
    while done==False:
        action=agent.take_action(state)
        next_state,reward,done,_=env.step(action)
        # print(next_state,reward,done)
        episode_return+=reward
        agent.update(state,action,reward,next_state)
        state=next_state
    return_list.append(episode_return)

print(f'回报：{return_list}')
print(f'最大回报：{max(return_list)}')

#训练完后执行
state=env.reset()
env.render()
done=False
while done==False:
    action=agent.take_action(state)
    next_state, reward, done, _=env.step(action)
    agent.update(state, action, reward, next_state)
    state=next_state
    env.render()

episode_list=list(range(len(return_list)))
plt.plot(episode_list,return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Q-learning on Cliff Walking')
plt.show()




