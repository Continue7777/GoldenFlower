#-*- coding:utf-8
import numpy as np
import tensorflow as tf
import random
import codecs

from GameEnv import GlodenFlower


class DQN:
    def __init__(self,embedding_size=10,sequence_length=20,learning_rate=0.01,batch_size=1000): #初始化
        self.embedding_size = embedding_size
        self.card_layer_unit = 20
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sigema = 0.5
        self.step = 0
        self.explore_alpha = 0.9 ** (self.step / 1000)

        self.actions_index_dicts = {"闷_2": 0, "闷_4": 1, "闷_8": 2, "闷开_0":3,"看_2": 4, "看_5": 5, "看_10": 6, "看_20": 7,
                               "开_0": 8, "丢_0": 9}
        self.action_reverse_index_dicts = {v: k for k, v in self.actions_index_dicts.items()}

        self.action_notsee_index_dicts = {"闷_2": 0, "闷_4": 1, "闷_8": 2, "闷开_0": 3 , "看" : 4}
        self.action_see_index_dicts = {"看_2": 0, "看_5": 1, "看_10": 2, "看_20": 3, "开_0": 4, "丢_0": 5}
        self.action_see_reverse_index_dicts = {v: k for k, v in self.action_see_index_dicts.items()}
        self.action_notsee_reverse_index_dicts = {v: k for k, v in self.action_notsee_index_dicts.items()}
        self.seq_action_index_dicts = {"A_闷_2": 0, "A_闷_4": 1, "A_闷_8": 2, "A_闷开_0": 3, "A_看_2": 4, "A_看_5": 5, "A_看_10": 6,
                                  "A_看_20": 7, "A_开_0": 8, "A_丢_0": 9,
                                  "B_闷_2": 10, "B_闷_4": 11, "B_闷_8": 12, "B_闷开_0": 13, "B_看_2": 14, "B_看_5": 15,
                                  "B_看_10": 16, "B_看_20": 17, "B_开_0": 18, "B_丢_0": 19}
        self.card_index_dicts = {'spade_1': 0, 'spade_2': 1, 'spade_3': 2, 'spade_4': 3, 'spade_5': 4, 'spade_6': 5,
                            'spade_7': 6, 'spade_8': 7, 'spade_9': 8, 'spade_10': 9, 'spade_J': 10, 'spade_Q': 11,
                            'spade_K': 12,
                            'heart_1': 13, 'heart_2': 14, 'heart_3': 15, 'heart_4': 16, 'heart_5': 17, 'heart_6': 18,
                            'heart_7': 19, 'heart_8': 20, 'heart_9': 21, 'heart_10': 22, 'heart_J': 23, 'heart_Q': 24,
                            'heart_K': 25,
                            'club_1': 26, 'club_2': 27, 'club_3': 28, 'club_4': 29, 'club_5': 30, 'club_6': 31,
                            'club_7': 32, 'club_8': 33, 'club_9': 34, 'club_10': 35, 'club_J': 36, 'club_Q': 37,
                            'club_K': 38,
                            'diamond_1': 39, 'diamond_2': 40, 'diamond_3': 41, 'diamond_4': 42, 'diamond_5': 43,
                            'diamond_6': 44, 'diamond_7': 45, 'diamond_8': 46, 'diamond_9': 47, 'diamond_10': 48,
                            'diamond_J': 49, 'diamond_Q': 50, 'diamond_K': 51," ":52}

        self.gameEnv = GlodenFlower([2000,2000])
        self.card_feature1_index_dicts = {self.gameEnv.scoreMap["豹子"]: 0, self.gameEnv.scoreMap["同花顺"]: 1, self.gameEnv.scoreMap["金花"]: 2,
                                          self.gameEnv.scoreMap["顺子"]: 3, self.gameEnv.scoreMap["对子"]: 4, self.gameEnv.scoreMap["单"]: 5}

        self.sess = tf.Session()
        self.build_network()
        self.sess.run(tf.global_variables_initializer())
        self.memory = []
        self.file = codecs.open("train_data.csv","w",encoding='utf-8')


    def get_weights(self,index_dicts, columns,embedding_size):
        res = {}
        for index_dict, column in zip(index_dicts, columns):
            res[column + "_emb"] = tf.Variable(tf.random_uniform([len(index_dict) + 1, embedding_size], -1.0, 1.0),
                                               name=column + "_emb")
        return res

    def collect_final_step_of_lstm(self,lstm_representation, lengths):
        # lstm_representation: [batch_size, passsage_length, dim]
        # lengths: [batch_size]
        lengths = tf.maximum(lengths, tf.zeros_like(lengths, dtype=tf.int32))

        batch_size = tf.shape(lengths)[0]
        batch_nums = tf.range(0, limit=batch_size)  # shape (batch_size)
        indices = tf.stack((batch_nums, lengths), axis=1)  # shape (batch_size, 2)
        result = tf.gather_nd(lstm_representation, indices, name='last-forwar-lstm')
        return result  # [batch_size, dim]

    def build_network(self): #构建网络模型
        self.weights = self.get_weights([self.seq_action_index_dicts,self.card_index_dicts,self.card_feature1_index_dicts],["seq_action","card","card_feature1"],self.embedding_size)

        self.global_steps = tf.Variable(0, trainable=False)
        self.playSequenceInput = tf.placeholder(shape=[None,self.sequence_length],dtype=tf.int32,name="playSequenceInput")
        self.personStatusInput = tf.placeholder(shape=[None,],dtype=tf.int32,name="personStatusInput") # 1:闷  0:看
        self.playSequenceLengthInput = tf.placeholder(shape=[None],dtype=tf.int32,name="playSequenceLengthInput")
        self.playCardsInput = tf.placeholder(shape=[None,3],dtype=tf.int32,name="playCardsInput")
        self.playCardsFeatureInput = tf.placeholder(shape=[None,1],dtype=tf.int32,name="playCardsInput")
        self.actionInput = tf.placeholder(shape=[None,len(self.actions_index_dicts)],dtype=tf.float32,name="actionInput")
        self.yInput = tf.placeholder(shape=[None,],dtype=tf.float32,name="yInput")
        self.mask = tf.constant([[0,0,0,0,1,1,1,1,1,1],[1, 1, 1, 1, 0, 0, 0, 0, 0, 0]],dtype=tf.float32)

        self.playSequenceEmb   = tf.nn.embedding_lookup(self.weights['seq_action_emb'], self.playSequenceInput) # bs * seq * emb
        self.playCardsEmb = tf.reshape(tf.nn.embedding_lookup(self.weights['card_emb'], self.playCardsInput),[-1, 3 * self.embedding_size]) # bs, 3 * emb
        self.playCardsFeatureEmb = tf.reshape(tf.nn.embedding_lookup(self.weights['card_feature1_emb'], self.playCardsFeatureInput),[-1,self.embedding_size])  # bs, 3 * emb

        cell = tf.nn.rnn_cell.LSTMCell(num_units=10, state_is_tuple=True)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell, cell_bw=cell, dtype=tf.float32, sequence_length=self.playSequenceLengthInput, inputs=self.playSequenceEmb
        )
        self.output_fw, self.output_bw = outputs
        self.last_output = self.collect_final_step_of_lstm(self.output_fw,self.playSequenceLengthInput-1)
        states_fw, states_bw = states

        card_layer1 = tf.layers.dense(tf.concat([self.playCardsEmb,self.playCardsFeatureEmb],1), self.card_layer_unit, activation=tf.nn.leaky_relu)
        card_layer = tf.layers.dense(card_layer1, int(self.card_layer_unit / 2), activation=tf.nn.leaky_relu)



        self.predictionsNotSee = tf.layers.dense(tf.nn.relu(tf.layers.dense(self.last_output, 10)),len(self.action_notsee_index_dicts)) # bs,notsee + 1
        self.predictionsSee = tf.layers.dense(tf.nn.relu(tf.layers.dense(tf.concat([self.last_output, card_layer], 1), 10)),len(self.action_see_index_dicts)) # bs,see
        self.prediction = tf.concat([self.predictionsNotSee[:,:-1],self.predictionsSee],1) # bs see+not_see
        self.maskOutput = tf.gather(self.mask,self.personStatusInput * tf.cast(~tf.equal(tf.arg_max(self.predictionsNotSee,1),len(self.action_notsee_index_dicts)),dtype=tf.int32))
        # 看 看 看 0
        # 看 闷 看 0
        # 闷 看 看 0
        # 闷 闷 闷 1
        self.predictions = self.prediction * self.maskOutput
        # self.predictions = tf.layers.dense(card_layer,len(self.actions_index_dicts),activation=tf.nn.leaky_relu)

        self.predictionsMaxQValue = tf.reduce_max(self.predictions)
        self.predictionsMaxQAction = tf.arg_max(self.predictions,1)

        # Get the predictions for the chosen actions only

        self.action_predictions =  tf.reduce_sum(tf.multiply(self.predictions, self.actionInput), reduction_indices=1)

        # Calculate the loss
        self.losses = tf.squared_difference(self.yInput, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_steps)

    def _feed_dict(self,status):
        statusMap = {"闷":1,"看":0,"开":0}
        playSequenceStr = status[:,0]
        playCardStr = status[:,1]
        personStatus = status[:,2]
        personIndex = [statusMap[i] for i in personStatus]
        playSequenceIndex = [[self.seq_action_index_dicts[i] for i in j][:20] + [len(self.seq_action_index_dicts)] * (len(self.seq_action_index_dicts) - len(j)) for j in playSequenceStr]
        playSequenceLength = [len(i)+1 for i in playSequenceStr]
        playCardIndex = [sorted([self.card_index_dicts[i] for i in j]) for j in playCardStr]
        playCardFeature = [[self.card_feature1_index_dicts[self.gameEnv.score(j)]] for j in playCardStr]
        return {self.playSequenceInput: np.array(playSequenceIndex), self.playCardsInput: np.array(playCardIndex),self.playSequenceLengthInput: np.array(playSequenceLength),
                self.personStatusInput:np.array(personIndex),self.playCardsFeatureInput:np.array(playCardFeature)}

    def get_max_Q(self,status):
        return self.sess.run(self.predictionsMaxQValue,feed_dict=self._feed_dict(status))

    def get_max_availble_action_value(self,status,personStatus,nowPrice):
        res = []
        _feed_dict = self._feed_dict(status)
        probs = self.sess.run(self.predictions, feed_dict=_feed_dict)
        for i in range(nowPrice.shape[0]):
            availble_actions = self.gameEnv._chooseAvailbleAction(personStatus[i],self.actions_index_dicts.keys(),nowPrice[i])
            avail_index_list = [self.actions_index_dicts[k] for k in availble_actions]
            res.append(max(probs[i,avail_index_list]))
        return res

    def get_action_Q(self,status,action): #通过训练好的网络，根据状态获取动作
        _feed_dict = self._feed_dict(status)
        _feed_dict[self.actionInput] = self._one_hot([self.actions_index_dicts[action]])
        return self.sess.run(self.action_predictions,feed_dict=self._feed_dict(status))

    def get_action_prob(self,status):
        _feed_dict = self._feed_dict(status)
        prob = self.sess.run(self.predictions, feed_dict=_feed_dict)
        res_dict = {k+self.action_reverse_index_dicts[k]:v for k,v in zip(self.action_reverse_index_dicts.keys(),prob[0])}
        return res_dict

    def choose_action(self,status,availble_actions): #通过训练好的网络，根据状态获取动作
        _feed_dict = self._feed_dict(status)
        prob = self.sess.run(self.predictions, feed_dict=_feed_dict)[0]
        max_value = -1000
        max_action = "丢_0"
        for action in availble_actions:
            if prob[self.actions_index_dicts[action]] > max_value:
                max_value = action

        if random.random() < max(0.9 ** (self.step / 500),0.05):
            return random.choice(availble_actions)
        return max_action

    def _one_hot(self,x):
        res = np.zeros((len(x), 10))
        res[[i for i in range(len(x))], x] = 1
        return res

    def train(self,train_data): #训练
        """
        memeory:[[ob_this,action,reward,done,ob_next],[ob_this...]]
        ob_this:[(seq,card,money),()]
        :return:
        """
        if train_data is None:
            train_data = self.experience_replay()
        train_observation_this = train_data[:,0]
        train_action =  train_data[:,1]
        train_reward = train_data[:,2]
        train_done = train_data[:,3]
        train_observation_next = train_data[:,4]
        Astatus = train_data[:, 5]
        now_price =  train_data[:, 6]

        statusMap = {"闷":1,"看":0,"开":0}
        playSequenceStr = [i[0] for i in train_observation_this]
        playCardStr =  [i[1] for i in train_observation_this]
        personStatus = [i[2] for i in train_observation_this]
        playSequenceIndex = [[self.seq_action_index_dicts[i] for i in j] + [len(self.seq_action_index_dicts)] * (len(self.seq_action_index_dicts) - len(j)) for j in playSequenceStr]
        playSequenceLength = [len(i)+1 for i in playSequenceStr]
        personIndex = [statusMap[i] for i in personStatus]
        playCardIndex = [sorted([self.card_index_dicts[i] for i in j]) for j in playCardStr]
        playCardFeature = [[self.card_feature1_index_dicts[self.gameEnv.score(j)]] for j in playCardStr]
        actionIndex = [self.actions_index_dicts[i] for i in  train_action]

        next_status = np.array([[i[0],i[1],i[2]] for i in train_observation_next])
        maxQNext = self.get_max_availble_action_value(next_status,Astatus,now_price)
        y = []
        for i in range(self.batch_size):
            if train_done[i] == True:
                y.append(train_reward[i])
            else:
                y.append(train_reward[i] + self.sigema * maxQNext[i])

        feed_dict = {self.playSequenceInput:np.array(playSequenceIndex),self.playCardsInput:np.array(playCardIndex),
                     self.actionInput:self._one_hot(actionIndex),self.playSequenceLengthInput:np.array(playSequenceLength),
                     self.yInput:np.array(y),self.personStatusInput:np.array(personIndex),self.playCardsFeatureInput:np.array(playCardFeature)}
        _, global_step,loss = self.sess.run([self.train_op, self.global_steps, self.loss], feed_dict=feed_dict)
        self.step = global_step
        if global_step % 100 == 0:
            print("loss",global_step,loss)

    # def save_model(self): #保存模型
    # def restore(self): #加载模型
    def store_transition(self,observation_this, action, reward,done,observation_next,Bcards,Astatus,now_price): #DQN存储记忆
        if len(observation_this[0]) < self.sequence_length:
            self.memory.append([observation_this,action,reward,done,observation_next,Astatus,now_price])
            self.file.write(str(observation_this) + "\t" + action + "\t" + str(reward) + "\t" + str(done) + "\t" + str(observation_next) + '\t' + str(Bcards)
                            + "\t" + str(Astatus) + "\t" + str(now_price) + "\n")
            if len(self.memory) > 10**7:
                self.memory = self.memory[:10*5]

    def experience_replay(self): #记忆回放
        return np.array(random.sample(self.memory, self.batch_size))

    def exerience_replay_final_step(self):
        data = np.array(self.memory)
        data = data[data[:, 2] != 0]
        return np.array(random.sample(data, self.batch_size))