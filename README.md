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
            observation:序列:a-action b-action ...   stop  A是agent,b是镜像agent
            reward:出牌没结束就是亏，出牌结束赢了就是赢之前桌子上的钱，每步单独一算
            done:是否结束
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

## 流程图
![NIPS 2013](https://img-blog.csdn.net/20170612221532013?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvdTAxMzIzNjk0Ng==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

## DQN伪码
    gameEnv = GlodenFlower([2000,2000])
    memory = []
    for episode in range(3):
        # 初始化环境
        gameEnv.reset()

    playerI = gameEnv.getStartTurn()
    while True:
        # DQN 根据观测值选择行为
        action = RL.choose_action(observation, playerI)
        # action = random.choice(gameEnv.chooseAvailbleAction(playerI))
        playerI = "B" if playerI == "A" else "A"

        # 环境根据行为给出下一个 state, reward, 是否终止
        observation_, reward, done = gameEnv.step(action, playerI)

        # DQN 存储记忆
        # RL.store_transition(observation, action, reward)
        memory.append((observation, action, reward))

        # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
        # if (step > 200) and (step % 5 == 0):
        #     RL.learn()

        # 将下一个 state_ 变为 下次循环的 state
        observation = observation_

        # 如果终止, 就跳出循环
        if done:
            break

        # end of game


# todo
√ 把demo改成强化学习环境的接口。
√ 编写DQN函数(采用2013版本)

# 实验记录
+ 当前maxQ+随机探索，每隔一段时间取历史数据训练。没用target网络。发现loss几乎没下降，需要检查下各个环境是否有bug。

# 实验待验证
+ 如果观察训练，DQN更新的是TD0误差，本身就不是真实值。如果探索合理的话，应该还是可以学到点东西？
+ target网络是否需要，进行下对比，网络是否存在不稳定情况。
+ 两个同模型同时进步，如何对比？
    + 采用第三方进行对比
    + 人工检验，看看学习到的策略是否合理
    + 固定某些状态，观察策略的变化
