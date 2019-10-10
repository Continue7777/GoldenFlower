#-*- coding:utf-8 -*-

import random
import copy
class GlodenFlower:
    def __init__(self,moneyList):
        self.debug = True

        # 游戏参数
        self.scoreMap = {"豹子":10,"同花顺":9,"金花":8,"顺子":7,"对子":6,"单":5}
        self.reverseScoreMap = {v: k for k, v in self.scoreMap.iteritems()}
        self.VMap = {"1":14,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10,"J":11,"Q":12,"K":13}
        self.gameStatsMap = {"on":1,"over":0}

        # 动作空间
        self.actionMoney = {"闷_2":-2,"闷_4":-4,"闷_8":-8,"看_2":-2,"看_5":-5,"看_10":-10,"看_20":-20,"闷开_1":0,"丢_1":0}

        # 全局状态
        self.personMoney = {"A":moneyList[0],"B":moneyList[1]}
        self.playerCards = {"A":"","B":""}

        # 每步参数
        self.whoWinLast = "A"
        self.deskMoney = 0
        self.nowPrice = 0
        self.gameStauts = self.gameStatsMap["on"]
        self.whosTurn = "A"
        self.personStatus = {"A":"闷","B":"闷"}
        self.playSequence = []


    def getStartTurn(self):
        return self.whoWinLast

    def getTurn(self):
        return self.whosTurn

    def reset(self):
        # 发牌
        self.deal()

        # 状态清零
        self.status_init()

        # 下底
        for playerI in self.personMoney.keys():
            self.personMoney[playerI] -= 1
        self.deskMoney += 2


    def stepA(self,action):
        observation_next,reward,done = self.step(action,"A")
        if done:
            return observation_next,reward,done
        action = random.choice(gameEnv.chooseAvailbleAction(playerI))
        print "B",action,self.deskMoney
        observation_next, reward, done = self.step(action,"B")
        if action == "丢_1":
            reward = self.deskMoney
        if done and reward == 0:
            reward = self.deskMoney
        return observation_next, reward, done


    def step(self,action,playerI):
        doneFlag = False
        giveupFlag = False
        self.playSequence.append(str(playerI) + "_" + action)
        observation = [copy.copy(self.playSequence),copy.copy(self.playerCards[playerI]),self.personMoney[playerI]]

        action_type,action_money = action.split("_")
        action_money = int(action_money)
        if action_type == "看" and self.nowPrice <= action_money:
            if action_money > self.personMoney[playerI]:
                raise Exception("没钱了1 Invalid !")
            else:
                self.personMoney[playerI] -= action_money
                self.deskMoney += action_money
                self.personStatus[playerI] = "看"
                self.nowPrice = action_money
        elif action_type == "闷" and self.nowPrice <= action_money * 2.5:
            if self.personStatus[playerI] == "看":
                raise Exception("已经看过了，不能闷了")
            if action_money > self.personMoney[playerI]:
                raise Exception("没钱了2 Invalid level!")
            else:
                self.personMoney[playerI] -= action_money
                self.deskMoney += action_money
                self.personStatus[playerI] = "闷"
                self.nowPrice = action_money * 2.5
        elif action_type == "开":
            if self.nowPrice > self.personMoney[playerI]:
                raise Exception("没钱了3 Invalid level!")
            else:
                self.personMoney[playerI] -= self.nowPrice
                self.deskMoney += self.nowPrice
                self.personStatus[playerI] = "开"
                self.gameStauts = self.gameStatsMap["over"]
                doneFlag = True
        elif action_type == "闷开":
            if self.nowPrice / 2.5 > self.personMoney[playerI]:
                raise Exception("没钱了4 Invalid level!")
            else:
                self.personMoney[playerI] -= self.nowPrice / 2.5
                self.deskMoney += self.nowPrice / 2.5
                self.personStatus[playerI] = "开"
                self.gameStauts = self.gameStatsMap["over"]
                doneFlag = True
        elif action_type == "丢":
            doneFlag = True
            giveupFlag = True
        else:
            raise Exception("异常操作！")

        reward = -action_money
        if doneFlag:
            AWin = self.compare(self.playerCards["A"], self.playerCards["B"])
            if AWin:
                self.whoWinLast = "A"
                self.personMoney["A"] += self.deskMoney
            else:
                self.whoWinLast = "B"
                self.personMoney["B"] += self.deskMoney
            if AWin and playerI == "A":
                reward = self.deskMoney
            elif AWin == False and playerI == "A":
                reward = 0
            elif AWin and playerI == "B":
                reward = 0
            elif AWin == False and playerI == "B":
                reward = self.deskMoney
        if giveupFlag:
            reward = 0
        return observation,reward,doneFlag

    def status_init(self):
        self.nowPrice = 2
        self.deskMoney = len(self.personMoney.keys())
        self.personStatus["A"] = "闷"
        self.personStatus["B"] = "闷"
        self.gameStauts = self.gameStatsMap["on"]
        self.playSequence = []

    def deal(self,n=2):
        cards = ['spade_1','spade_2','spade_3','spade_4','spade_5','spade_6','spade_7','spade_8','spade_9','spade_10','spade_J','spade_Q','spade_K',
                'heart_1', 'heart_2', 'heart_3', 'heart_4', 'heart_5', 'heart_6', 'heart_7', 'heart_8', 'heart_9','heart_10', 'heart_J', 'heart_Q', 'heart_K',
                'club_1', 'club_2', 'club_3', 'club_4', 'club_5', 'club_6', 'club_7', 'club_8', 'club_9', 'club_10','club_J', 'club_Q', 'club_K',
                'diamond_1', 'diamond_2', 'diamond_3', 'diamond_4', 'diamond_5', 'diamond_6', 'diamond_7', 'diamond_8','diamond_9', 'diamond_10', 'diamond_J', 'diamond_Q', 'diamond_K']

        player1_card = [cards.pop(random.randrange(len(cards))) for i in range(3)]
        player2_card = [cards.pop(random.randrange(len(cards))) for i in range(3)]
        self.playerCards['A'] = player1_card
        self.playerCards['B'] = player2_card


    def score(self,cards):
        k1, v1 = cards[0].split("_")
        k2, v2 = cards[1].split("_")
        k3, v3 = cards[2].split("_")
        v1 = self.VMap[v1]
        v2 = self.VMap[v2]
        v3 = self.VMap[v3]
        v1,v2,v3 = sorted([v1,v2,v3])

        if v1 == v2 and v1 == v3:
            return self.scoreMap['豹子']
        elif k1 == k2 and k1 == k3:
            if v1 + 1 == v2 and  v2 + 1 == v3:
                return self.scoreMap["同花顺"]
            else:
                return self.scoreMap["金花"]
        elif   v1 + 1 == v2 and  v2 + 1 == v3:
            return self.scoreMap["顺子"]
        elif v1 == v2 or v1 == v3 or v2 == v3:
            return self.scoreMap["对子"]
        else:
            return self.scoreMap["单"]

    def compareSingle(self,vA,vB):
        if vA[0] > vB[0]:
            return True
        elif vA[0] < vB[0]:
            return False
        else:
            if vA[1] > vB[1]:
                return True
            elif vA[1] < vB[1]:
                return False
            else:
                if vA[2] > vB[2]:
                    return True
                elif vA[2] < vB[2]:
                    return False
        return "="

    def compare(self,cardsA,cardsB):
        firstScoreA = self.score(cardsA)
        firstScoreB = self.score(cardsB)

        if self.debug:print self.reverseScoreMap[firstScoreA],self.reverseScoreMap[firstScoreB]

        vA = sorted([self.VMap[i.split("_")[1]] for i in cardsA],reverse=True)
        vB = sorted([self.VMap[i.split("_")[1]] for i in cardsB],reverse=True)

        if firstScoreA > firstScoreB:
            return True
        elif firstScoreA < firstScoreB:
            return False
        else:
            return self.compareSingle(vA,vB)



    def chooseAvailbleAction(self,playerI):
        curStatus = self.personStatus[playerI]
        res = []
        for action in self.actionMoney.keys():
            action_type = action.split("_")[0]
            action_value = int(action.split("_")[1])
            if action_type == "开" or action_type == "丢" :res.append(action)
            if curStatus == "看" and action_type == "闷":continue
            if curStatus == "看":
                if self.nowPrice > action_value:continue
            elif curStatus == "闷":
                res.append("闷开_1")
                if self.nowPrice  > 2.5 * action_value: continue

            res.append(action)
        return res



gameEnv = GlodenFlower([2000,2000])
memory = []
for episode in range(3):
    # 初始化环境
    gameEnv.reset()

    # playerI = gameEnv.getStartTurn()
    playerI = "A"
    observation_this = []
    while True:
        # DQN 根据观测值选择行为
        # action = RL.choose_action(observation_this, playerI)
        # if playerI == "B":
        #     action = random.choice(gameEnv.chooseAvailbleAction(playerI))
        #     gameEnv.step(action,"B")
        #     print playerI, action, gameEnv.deskMoney, gameEnv.nowPrice


        # 环境根据行为给出下一个 state, reward, 是否终止
        action = random.choice(gameEnv.chooseAvailbleAction(playerI))
        print playerI, action, gameEnv.deskMoney, gameEnv.nowPrice
        observation_next, reward, done = gameEnv.stepA(action)


        # DQN 存储记忆
        # RL.store_transition(observation, action, reward)
        memory.append((observation_this, action, reward,observation_next))

        # 控制学习起始时间和频率 (先累积一些记忆再开始学习)
        # if (step > 200) and (step % 5 == 0):
        #     RL.learn()

        # 将下一个 state_ 变为 下次循环的 state
        observation_this = observation_next

        # 如果终止, 就跳出循环
        if done:
            break

        # end of game
print('game over')
for i in memory:
    print i[0],"\taction:",i[1],"\treward:",i[2],i[3]