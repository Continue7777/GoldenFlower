#-*- coding:utf-8
import numpy as np
import tensorflow as tf
import random
import codecs

class DQN:
    def __init__(self,embedding_size=10,sequence_length=20,learning_rate=0.01,batch_size=1000): #初始化
        self.embedding_size = embedding_size
        self.card_layer_unit = 10
        self.sequence_length = sequence_length
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.sigema = 0.9
        self.step = 0
        self.explore_alpha = 0.9 ** (self.step / 1000)

        self.actions_index_dicts = {"闷_2": 0, "闷_4": 1, "闷_8": 2, "看_2": 3, "看_5": 4, "看_10": 5, "看_20": 6, "闷开_0": 7,
                               "开_0": 8, "丢_0": 9}
        self.action_reverse_index_dicts = {v: k for k, v in self.actions_index_dicts.items()}
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

    def build_network(self): #构建网络模型
        weights = self.get_weights([self.seq_action_index_dicts,self.card_index_dicts],["seq_action","card"],self.embedding_size)

        self.global_steps = tf.Variable(0, trainable=False)
        self.playSequenceInput = tf.placeholder(shape=[None,self.sequence_length],dtype=tf.int32,name="playSequenceInput")
        self.playSequenceLengthInput = tf.placeholder(shape=[None],dtype=tf.int32,name="playSequenceLengthInput")
        self.playCardsInput = tf.placeholder(shape=[None,3],dtype=tf.int32,name="playCardsInput")
        self.actionInput = tf.placeholder(shape=[None,1],dtype=tf.int32,name="actionInput")
        self.yInput = tf.placeholder(shape=[None,1],dtype=tf.float32,name="yInput")

        self.playSequenceEmb   = tf.nn.embedding_lookup(weights['seq_action_emb'], self.playSequenceInput) # bs * seq * emb
        self.playCardsEmb = tf.reshape(tf.nn.embedding_lookup(weights['card_emb'], self.playCardsInput),[-1, 3 * self.embedding_size]) # bs, 3 * emb

        cell = tf.nn.rnn_cell.LSTMCell(num_units=10, state_is_tuple=True)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=cell, cell_bw=cell, dtype=tf.float32, sequence_length=self.playSequenceLengthInput, inputs=self.playSequenceEmb
        )
        output_fw, output_bw = outputs
        states_fw, states_bw = states

        card_layer = tf.layers.dense(self.playCardsEmb, self.card_layer_unit,activation=tf.nn.leaky_relu)

        self.predictions = tf.layers.dense(tf.concat([output_fw[:,-1,:], card_layer],1), len(self.actions_index_dicts))
        self.predictionsMaxQValue = tf.reduce_max(self.predictions)
        self.predictionsMaxQAction = tf.arg_max(self.predictions,1)

        # Get the predictions for the chosen actions only
        gather_indices = tf.range(self.batch_size) * len(self.actions_index_dicts) + self.actionInput
        self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)

        # Calculate the loss
        self.losses = tf.squared_difference(self.yInput, self.action_predictions)
        self.loss = tf.reduce_mean(self.losses)

        # Optimizer Parameters from original paper
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self.global_steps)

    def _feed_dict(self,status):
        playSequenceStr = status[:,0]
        playCardStr = status[:,1]
        playSequenceIndex = [[self.seq_action_index_dicts[i] for i in j][:20] + [len(self.seq_action_index_dicts)] * (len(self.seq_action_index_dicts) - len(j)) for j in playSequenceStr]
        playSequenceLength = [len(i)+1 for i in playSequenceStr]
        playCardIndex = [[self.card_index_dicts[i] for i in j] for j in playCardStr]
        return {self.playSequenceInput: np.array(playSequenceIndex), self.playCardsInput: np.array(playCardIndex),self.playSequenceLengthInput: np.array(playSequenceLength)}

    def get_max_Q(self,status):
        return self.sess.run(self.predictionsMaxQValue,feed_dict=self._feed_dict(status))

    def get_max_action(self,status):
        return self.sess.run(self.predictionsMaxQAction,feed_dict=self._feed_dict(status))

    def get_action_Q(self,status,action): #通过训练好的网络，根据状态获取动作
        _feed_dict = self._feed_dict(status)
        _feed_dict[self.actionInput] = np.array(self.actions_index_dicts[action])
        return self.sess.run(self.action_predictions,feed_dict=self._feed_dict(status))

    def choose_action(self,status,availble_actions): #通过训练好的网络，根据状态获取动作
        action_index = self.get_max_action(status)
        if action_index not in availble_actions:
            return random.choice(availble_actions)
        else:
            if random.random() < 0.9 ** (self.step / 500):
                return random.choice(availble_actions)
        return self.action_reverse_index_dicts[action_index[0]]

    def train(self): #训练
        """
        memeory:[[ob_this,action,reward,done,ob_next],[ob_this...]]
        ob_this:[(seq,card,money),()]
        :return:
        """
        train_data = self.experience_replay()
        train_observation_this = train_data[:,0]
        train_action =  train_data[:,1]
        train_reward = train_data[:,2]
        train_done = train_data[:,3]
        train_observation_next = train_data[:,4]

        playSequenceStr = [i[0] for i in train_observation_this]
        playCardStr =  [i[1] for i in train_observation_this]
        playSequenceIndex = [[self.seq_action_index_dicts[i] for i in j] + [len(self.seq_action_index_dicts)] * (len(self.seq_action_index_dicts) - len(j)) for j in playSequenceStr]
        playSequenceLength = [len(i)+1 for i in playSequenceStr]
        playCardIndex = [[self.card_index_dicts[i] for i in j] for j in playCardStr]
        actionIndex = [self.actions_index_dicts[i] for i in  train_action]

        next_status = np.array([[i[0],i[1],i[2]] for i in train_observation_next])
        maxQNext = self.get_max_action(next_status)
        y = []
        for i in range(self.batch_size):
            if train_done[i] == True:
                y.append(train_reward[i])
            else:
                y.append(train_reward[i] + self.sigema * maxQNext[i])

        feed_dict = {self.playSequenceInput:np.array(playSequenceIndex),self.playCardsInput:np.array(playCardIndex),
                     self.actionInput:np.array(actionIndex).reshape(self.batch_size,-1),self.playSequenceLengthInput:np.array(playSequenceLength),
                     self.yInput:np.array(y).reshape(self.batch_size,-1)}
        _, global_step,loss = self.sess.run([self.train_op, self.global_steps, self.loss], feed_dict=feed_dict)
        self.step = global_step
        if global_step % 100 == 0:
            print("loss",global_step,loss)

    # def save_model(self): #保存模型
    # def restore(self): #加载模型
    def store_transition(self,observation_this, action, reward,done,observation_next): #DQN存储记忆
        if len(observation_this[0]) < self.sequence_length:
            self.memory.append([observation_this,action,reward,done,observation_next])
            self.file.write(str(observation_this) + "\t" + action + "\t" + str(reward) + "\t" + str(done) + "\t" + str(observation_next) + "\n")
            if len(self.memory) > 10**7:
                self.memory = self.memory[:10*5]

    def experience_replay(self): #记忆回放
        return np.array(random.sample(self.memory, self.batch_size))