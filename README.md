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
+ 奖励：闷、看都是0、开看输赢加减钱。
+ 状态：本局动作序列,自己手牌
+ 行动策略：e-greedy探索

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
+ 步数放大之后，开始下降，后来抖动，调整随机参数，继续测试，并验证下中间结果是否有误。

# 问题解决
+ 问: loss不下降，预测值和牌的大小关系不太符合。
+ 解：
    + 修复env bug
    + train的频率提起来
    + 调整下学习率
    + 增加卡牌特征，牌的可能性在10w级别，现在训练较慢，想快速看到效果，就舍弃一下端到端，手工加一些强特征。
+ 结论：
    + 以上操作后，终点节点数据已经接近预期，下一步动手处理延迟奖励的传播问题。（至此，做的事情就是一个简单的统计赢的概率而已，但是这是DQN迭代公式生效的基础）
    + 明显可以看到现在数据的探索性很差，接近尾声的时候大概率丢牌，所以"豹子" 、 "金花"并没有区别，这里就额外显得探索的重要性。在数据量有限的情况下，尽可能探索出合理的
    游戏过程，能够大大加速智能体的进化。豹子和金花如果都是遇到丢牌就难以比较，容易淹没在数据中，但是如果都高置信的，就能一直杠上，这样这个点就被突出起来，有可能被网络捕捉。学到差异。

# 总结
该游戏本身机制存在问题，不做更多的探索：
+ 当钱无限多的时候，闷着就赢，谁看谁输，显然模型学到了这个特性，这个游戏已经失去了意义。
+ 如果大家都很贪，希望得到更多的收益，先摸清对方想法的收益更大。
+ 如果一个明着，一个闷着，闷着的第一手开牌，自己纯随机，对方就没有太多操作空间了。这里做一个简单的解释，两个人玩石头剪刀布，一个人纯随机出，胜率就都是50%，如果一个人有偏向，另外的一个人
就可以抓住机会，可能越会找规律的胜率越高，双方是一个动态博弈的过程。
