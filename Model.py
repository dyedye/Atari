import tensorflow as tf
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense
from collections import deque
import numpy as np
from skimage 

INITIAL_EPSILON = 1.0
FINAL_EPSILON = 0.1
EXPLORATION_STEPS = 1000000
LEARNING_RATE = 0.00025  # RMSPropで使われる学習率
MOMENTUM = 0.95  # RMSPropで使われるモメンタム
MIN_GRAD = 0.01  # RMSPropで使われる0で割るのを防ぐための値

class DQN_Agent():
    def __init__(self, num_actions):
        self.num_actions = num_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.time_step = 0
        self.repeated_action = 0

        # Initialize Q Network
        self.replay_memory = deque()
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Initialize Target Network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        #Configure pipeline to periodically update Target Network
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i])for i in range(len(target_network.trainable))]

        #Configure pipeline to train the model
        self.a, self.y, self.loss, self.grad_update = self.build_training_op(q_network_weights)

        #Configure Session
        self.sess = tf.InteractiveSession()

        #Initialize Variables
        self.sess.run(tf.initialize_all_variables())

        #Initialize Target Network
        self.sess.run(self.update_target_network)

        def build_network(self):
            model = Sequential()
            model.add(Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
            model.add(Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu'))
            model.add(Convolution2D(64, 3, 3, subsample=(1, 1), activation='relu'))
            model.add(Flatten())
            model.add(Dense(512, activation='relu'))
            model.add(Dense(self.num_actions))

            s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
            q_values = model(s)

            return s, q_values, model

        def build_training_op(self, q_network_weights):
            a = tf.placeholder(tf.int64, [None])
            y = tf.placeholder(tf.float32, [None])

            a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
            q_value = tf.reduce_sum(tf.mul(self.q_values, a_one_hot), reduction_indices=1)
            
            #Error Clipping
            error = tf.abs(y - q_value)
            quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
            linear_part = error - quadratic_part
            loss = tf.reduce_mean(0.5*tf.square(quadratic_part) + linear_part)
            optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM)
            grad_update = optimizer.minimize(loss, var_list=q_network_weights)

            return a, y, grad_update
        
        def get_initial_state(self, observation, last_observation):
            processed_observation = np.maximum(observation, last_observation)
