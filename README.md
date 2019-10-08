# GoldenFlower
强化学习扎金花

# 版本说明
构建双人对战基本逻辑：
动作集合：{"闷_2":-2,"闷_4":-4,"闷_8":-8,"看_2":-2,"看_5":-5,"看_10":-10,"看_20":-20,"闷开_1":0,"丢_1":0}

函数说明：
chooseAvailbleAction 获取某人的合理动作空间

# 迭代说明
构想了三种方案：
+ DQN，值函数。
+ 蒙特卡洛法，类似alphaGoZero。
+ AC框架下DDPG或者PPO，策略和值函数都单独走。

# DQN
+ 动作空间：闷_2、闷_4、闷_8、闷开、看_2、看_5、看_10、看_20、开、丢
+ 奖励：闷、看都是减钱、开看输赢加减钱，输光给个很大的负值。
+ 状态：本局动作序列，自己的本金，自己手牌

## env接口
    class GoldenFlower:
        def __init__()： #初始化
        def reset()： #初始化环境
        def step(self,action):return ovbservation,reward,done #环境交互
       
## DQN接口
    class DQN：
        def __init__(): #初始化
        def build_network(): #构建网络模型
        def get_action(self,status) #通过训练好的网络，根据状态获取动作
        def save_model() #保存模型
        def restore() #加载模型
        def store_transition(observation, action, reward, observation_) #DQN存储记忆
        def experience_replay() #记忆回放
        def train() #训练

## DQN伪码
    def run_maze():
        step = 0    # 用来控制什么时候学习
        for episode in range(300):
            # 初始化环境
            observation = env.reset()

            while True:
                # 刷新环境
                env.render()

                # DQN 根据观测值选择行为
                action = RL.choose_action(observation)

                # 环境根据行为给出下一个 state, reward, 是否终止
                observation_, reward, done = env.step(action)

                # DQN 存储记忆
                RL.store_transition(observation, action, reward, observation_)

                # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
                if (step > 200) and (step % 5 == 0):
                    RL.learn()

                # 将下一个 state_ 变为 下次循环的 state
                observation = observation_

                # 如果终止, 就跳出循环
                if done:
                    break
                step += 1   # 总步数

        # end of game
        print('game over')
        env.destroy()

 
    if __name__ == "__main__":
        env = Maze()
        RL = DeepQNetwork(env.n_actions, env.n_features,
                          learning_rate=0.01,
                          reward_decay=0.9,
                          e_greedy=0.9,
                          replace_target_iter=200,  # 每 200 步替换一次 target_net 的参数
                          memory_size=2000, # 记忆上限
                          # output_graph=True   # 是否输出 tensorboard 文件
                          )
        env.after(100, run_maze)
        env.mainloop()
        RL.plot_cost()  # 观看神经网络的误差曲线

# todo
+ 把demo改成强化学习环境的接口。
