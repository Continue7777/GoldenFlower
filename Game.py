#-*- coding:utf-8 -*-

import random
class GlodenFlower:
    def __init__(self,moneyList):
        # 豹子 > 同花顺 > 金花 > 顺子 > 对子 > 单
        self.scoreMap = {"豹子":10,"同花顺":9,"金花":8,"顺子":7,"对子":6,"单":5}
        self.gameStatsMap = {"on":1,"over":0}
        self.reverseScoreMap = {v: k for k, v in self.scoreMap.iteritems()}
        self.VMap = {"1":14,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9,"10":10,"J":11,"Q":12,"K":13}
        self.debug = True
        self.personMoney = {"A":moneyList[0],"B":moneyList[1]}
        self.deskMoney = 0
        self.personStatus = {"A":"闷","B":"闷"}
        self.actionMoney = {"闷_2":-2,"闷_4":-4,"闷_8":-8,"看_2":-2,"看_5":-5,"看_10":-10,"看_20":-20,"闷开_1":0,"丢_1":0}
        self.nowPrice = 0
        self.gameStauts = self.gameStatsMap["on"]

    def deal(self,n=2):
        cards = ['spade_1','spade_2','spade_3','spade_4','spade_5','spade_6','spade_7','spade_8','spade_9','spade_10','spade_J','spade_Q','spade_K',
                'heart_1', 'heart_2', 'heart_3', 'heart_4', 'heart_5', 'heart_6', 'heart_7', 'heart_8', 'heart_9','heart_10', 'heart_J', 'heart_Q', 'heart_K',
                'club_1', 'club_2', 'club_3', 'club_4', 'club_5', 'club_6', 'club_7', 'club_8', 'club_9', 'club_10','club_J', 'club_Q', 'club_K',
                'diamond_1', 'diamond_2', 'diamond_3', 'diamond_4', 'diamond_5', 'diamond_6', 'diamond_7', 'diamond_8','diamond_9', 'diamond_10', 'diamond_J', 'diamond_Q', 'diamond_K']

        player1_card = [cards.pop(random.randrange(len(cards))) for i in range(3)]
        player2_card = [cards.pop(random.randrange(len(cards))) for i in range(3)]
        return player1_card,player2_card

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

    def action(self,playerI,action):
        action_type,action_money = action.split("_")
        action_money = int(action_money)
        if action_type == "看" and self.nowPrice <= action_money:
            if action_money > self.personMoney[playerI]:
                print playerI," 没钱了1"
                return False
            else:
                self.personMoney[playerI] -= action_money
                self.deskMoney += action_money
                self.personStatus[playerI] = "看"
                self.nowPrice = action_money
        elif action_type == "闷" and self.nowPrice <= action_money * 2.5:
            if self.personStatus[playerI] == "看":
                print playerI," 已经看过了，不能闷了"
                return False
            if action_money > self.personMoney[playerI]:
                print playerI," 没钱了2"
                return False
            else:
                self.personMoney[playerI] -= action_money
                self.deskMoney += action_money
                self.personStatus[playerI] = "闷"
                self.nowPrice = action_money * 2.5
        elif action_type == "开":
            if self.nowPrice > self.personMoney[playerI]:
                print playerI," 没钱了3"
                return False
            else:
                self.personMoney[playerI] -= self.nowPrice
                self.deskMoney += self.nowPrice
                self.personStatus[playerI] = "开"
                self.gameStauts = self.gameStatsMap["over"]
        elif action_type == "闷开":
            if self.nowPrice / 2.5 > self.personMoney[playerI]:
                print playerI," 没钱了3"
                return False
            else:
                self.personMoney[playerI] -= self.nowPrice / 2.5
                self.deskMoney += self.nowPrice / 2.5
                self.personStatus[playerI] = "开"
                self.gameStauts = self.gameStatsMap["over"]
        return True

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

    def play(self,n):
        player1Cards,player2Cards = self.deal(n)
        if self.debug:print player1Cards,player2Cards
        while 1:
            for playerI in ["A","B"]:
                action = random.choice(self.chooseAvailbleAction(playerI))
                self.action(playerI,action)
                if self.debug:
                    print playerI,action,self.deskMoney,self.nowPrice

                if action == "丢_1":
                    if playerI == "A":
                        self.personMoney["B"] += self.deskMoney
                    elif playerI == "B":
                        self.personMoney["A"] += self.deskMoney
                    return

                if self.gameStauts == self.gameStatsMap["over"]:
                    flag = self.compare(player1Cards, player2Cards)
                    if flag:
                        self.personMoney["A"] += self.deskMoney
                    elif flag == False:
                        self.personMoney["B"] += self.deskMoney
                    else:
                        if playerI == "A":
                            self.personMoney["B"] += self.deskMoney
                        else:
                            self.personMoney["A"] += self.deskMoney
                    return

    def init(self):
        self.nowPrice = 2
        self.deskMoney = len(self.personMoney.keys())
        for playerI in self.personMoney.keys():
            self.personMoney[playerI] -= 1
        self.gameStauts = self.gameStatsMap["on"]

game = GlodenFlower([2000,2000])
for i in range(100):
    game.init()
    flag = game.play(2)
    print game.personMoney
    print
