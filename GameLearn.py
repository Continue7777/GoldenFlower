import GameEnv
import random
import numpy as np
import DQN
if __name__ == '__main__':
    gameEnv = GameEnv.GlodenFlower([20000,20000])
    RLModel = DQN.DQN(embedding_size=10,sequence_length=20,learning_rate=0.0001,batch_size=500)
    gameEnv.debug = False
    memory = []
    for episode in range(100000):
        # 初始化环境
        gameEnv.reset()

        playerI = gameEnv.getStartTurn()
        #print (playerI,"win last")
        observation_this = [[],gameEnv.playerCards["A"],gameEnv.personMoney["A"]]
        if playerI == "B":
            availble_actions = gameEnv.chooseAvailbleAction(playerI)
            action = RLModel.choose_action(np.array([observation_this]),availble_actions)
            #print (playerI, action, gameEnv.deskMoney, gameEnv.nowPrice)
            observation_next, reward, done = gameEnv.step(action, "B")
            playerI = "A"
            if done:
                continue

        while True:
            # DQN 根据观测值选择行为
            availble_actions = gameEnv.chooseAvailbleAction(playerI)
            action = RLModel.choose_action(np.array([observation_this]),availble_actions)
            # 环境根据行为给出下一个 state, reward, 是否终止
            #print (playerI, action, gameEnv.deskMoney, gameEnv.nowPrice,gameEnv.personStatus[playerI])
            observation_next, reward, done = gameEnv.stepA(action,RLModel)


            # DQN 存储记忆
            # RL.store_transition(observation, action, reward)
            RLModel.store_transition(observation_this, action, reward,done,observation_next)

            # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
            if (episode > 2000) and (episode % 10 == 0):
                RLModel.train()

            # 将下一个 state_ 变为 下次循环的 state
            observation_this = observation_next

            # 如果终止, 就跳出循环
            if done:
                break

            # end of game
    print('game over')

