#-*- coding:utf-8
import random
import copy
import numpy as np
class GlodenFlower:
    def __init__(self,moneyList):
        self.debug = True

        # 游戏参数
        self.scoreMap = {"豹子":10,"同花顺":9,"金花":8,"顺子":7,"对子":6,"单":5}
        self.reverseScoreMap = {v: k for k, v in self.scoreMap.items()}
        self.VMap = {"1":14,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10,"J":11,"Q":12,"K":13}
        self.gameStatsMap = {"on":1,"over":0}

        # 动作空间
        self.actionMoney = {"闷_2":-2,"闷_4":-4,"闷_8":-8,"看_2":-2,"看_5":-5,"看_10":-10,"看_20":-20,"开_0":0,"闷开_0":0,"丢_0":0}

        # 全局状态
        self.personMoney = {"A":moneyList[0],"B":moneyList[1]}
        self.playerCards = {"A":"","B":""}
        self.stepsNum = 0

        # 每步参数
        self.whoWinLast = "A"
        self.nowPrice = 0
        self.gameStauts = self.gameStatsMap["on"]
        self.whosTurn = "A"
        self.personPayed = {"A":0,"B":0}
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
        self.personPayed["A"] = 1
        self.personPayed["B"] = 1



    def stepA(self,action,RLModel):
        observation_next,reward,done = self.step(action,"A")
        if done:
            return observation_next,reward,done
        if RLModel:
            availble_actions = self.chooseAvailbleAction("B")
            actionB = RLModel.choose_action(np.array([observation_next]),availble_actions)
        else:
            actionB = random.choice(self.chooseAvailbleAction("B"))
        if self.debug:print ("player:%s ,action:%s A_pay:%s B_pay:%s nowPrice:%s" % ("B", actionB, gameEnv.personPayed["A"] ,gameEnv.personPayed["B"], gameEnv.nowPrice))
        observation_next, rewardB, done = self.step(actionB,"B")
        if actionB == "丢_0": # 对手弃牌
            reward = self.personPayed["B"]
        elif done and rewardB == 0: # 对手开牌输了
            reward = self.personPayed["B"]
        elif done:
            reward = -self.personPayed["A"]
        return observation_next, reward, done


    def step(self,action,playerI):
        self.stepNum += 1
        doneFlag = False
        giveupFlag = False
        self.playSequence.append(str(playerI) + "_" + action)

        action_type,action_money = action.split("_")
        action_money = int(action_money)
        if action_type == "看" and self.nowPrice <= action_money:
            if action_money > self.personMoney[playerI]:
                raise Exception("没钱了1 Invalid !")
            else:
                self.personMoney[playerI] -= action_money
                self.personPayed[playerI] += action_money
                self.personStatus[playerI] = "看"
                self.nowPrice = action_money
        elif action_type == "闷" and self.nowPrice <= action_money * 2.5:
            if self.personStatus[playerI] == "看":
                raise Exception("已经看过了，不能闷了")
            if action_money > self.personMoney[playerI]:
                raise Exception("没钱了2 Invalid level!")
            else:
                self.personMoney[playerI] -= action_money
                self.personPayed[playerI] += action_money
                self.personStatus[playerI] = "闷"
                self.nowPrice = action_money * 2.5
        elif action_type == "开":
            if self.nowPrice > self.personMoney[playerI]:
                raise Exception("没钱了3 Invalid level!")
            else:
                self.personMoney[playerI] -= self.nowPrice
                self.personPayed[playerI] += self.nowPrice
                self.personStatus[playerI] = "开"
                self.gameStauts = self.gameStatsMap["over"]
                doneFlag = True
        elif action_type == "闷开":
            if self.nowPrice / 2.5 > self.personMoney[playerI]:
                raise Exception("没钱了4 Invalid level!")
            else:

                self.personMoney[playerI] -= max(self.nowPrice / 2.5,1)
                self.personPayed[playerI] += max(self.nowPrice / 2.5,1)
                self.personStatus[playerI] = "开"
                self.gameStauts = self.gameStatsMap["over"]
                doneFlag = True
        elif action_type == "丢":
            doneFlag = True
            giveupFlag = True
        else:
            raise Exception("异常操作！",action)

        reward = 0
        if doneFlag:
            AWin = self.compare(self.playerCards["A"], self.playerCards["B"])
            if playerI == "A" and action_type == "丢":
                AWin = False
            if playerI == "B" and action_type == "丢":
                AWin = True
            if AWin:
                self.whoWinLast = "A"
                self.personMoney["A"] += (self.personPayed['A'] + self.personPayed["B"])
            else:
                self.whoWinLast = "B"
                self.personMoney["B"] += (self.personPayed['A'] + self.personPayed["B"])
            if AWin and playerI == "A":
                reward = self.personPayed["B"]
            elif AWin == False and playerI == "A":
                reward = -self.personPayed["A"]
            elif AWin and playerI == "B":
                reward = -self.personPayed["B"]
            elif AWin == False and playerI == "B":
                reward = self.personPayed["A"]

        observation = [copy.copy(self.playSequence),copy.copy(self.playerCards["A"]),self.personStatus["A"]]
        return observation,reward,doneFlag

    def status_init(self):
        self.stepNum = 0
        self.nowPrice = 2
        self.personStatus["A"] = "闷"
        self.personStatus["B"] = "闷"
        self.gameStauts = self.gameStatsMap["on"]
        self.personPayed["A"] = 0
        self.personPayed["B"] = 0
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
        elif v1 + 1 == v2 and  v2 + 1 == v3:
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

        if self.debug:print (self.reverseScoreMap[firstScoreA],self.reverseScoreMap[firstScoreB],self.playerCards["A"],self.playerCards["B"])

        vA = sorted([self.VMap[i.split("_")[1]] for i in cardsA],reverse=True)
        vB = sorted([self.VMap[i.split("_")[1]] for i in cardsB],reverse=True)

        if firstScoreA > firstScoreB:
            return True
        elif firstScoreA < firstScoreB:
            return False
        else:
            return self.compareSingle(vA,vB)



    def _chooseAvailbleAction(self,personStatus,actions,nowPrice):
        curStatus = personStatus
        res = []
        for action in actions:
            action_type = action.split("_")[0]
            action_value = int(action.split("_")[1])
            if curStatus == "看" and action_type == "开":
                pass
            elif curStatus == "闷" and action_type == "闷开":
                pass
            elif action_type == "丢":
                pass
            elif curStatus == "看" and action_type == "闷":
                continue
            elif curStatus == "看" and action_type == "看":
                if nowPrice > action_value: continue
            elif curStatus == "闷" and action_type == "看":
                if nowPrice > action_value: continue
            elif curStatus == "闷" and action_type == "闷":
                if nowPrice > 2.5 * action_value: continue
            else:
                continue
            res.append(action)
        return res

    def chooseAvailbleAction(self,playerI):
        return self._chooseAvailbleAction(self.personStatus[playerI],self.actionMoney.keys(),self.nowPrice)


if __name__ == '__main__':

    gameEnv = GlodenFlower([2000,2000])
    memory = []
    for episode in range(1000):
        # 初始化环境
        gameEnv.reset()

        playerI = gameEnv.getStartTurn()
        print (playerI,"win last")
        observation_this = [[], gameEnv.playerCards["A"], gameEnv.personStatus["A"]]
        if playerI == "B":
            action = random.choice(gameEnv.chooseAvailbleAction(playerI))
            print ("player:%s ,action:%s A_pay:%s B_pay:%s nowPrice:%s A status:%s" % (playerI, action, gameEnv.personPayed["A"] ,gameEnv.personPayed["B"], gameEnv.nowPrice,gameEnv.personStatus["A"]))
            observation_next, reward, done = gameEnv.step(action, "B")
            playerI = "A"
            if done:
                continue

        while True:
            # DQN 根据观测值选择行为
            # action = RL.choose_action(observation_this, playerI)
            # 环境根据行为给出下一个 state, reward, 是否终止
            actions = gameEnv.chooseAvailbleAction(playerI)
            action = random.choice(actions)
            print ("player:%s ,action:%s A_pay:%s B_pay:%s nowPrice:%s A status:%s" % (playerI, action, gameEnv.personPayed["A"] ,gameEnv.personPayed["B"], gameEnv.nowPrice,gameEnv.personStatus["A"]))
            observation_next, reward, done = gameEnv.stepA(action,None)


            # DQN 存储记忆
            # RL.store_transition(observation, action, reward)
            memory.append((observation_this, action, reward,done,observation_next))

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
    print (len(memory))
    for i in memory:
        print (i[0],"\taction:",i[1],"\treward:",i[2],i[3])

