import gym
import numpy as np
import random
import preprocessing
import tensorflow as tf
from keras import layers
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Model

from collections import deque
from keras.optimizers import RMSprop
from keras import backend as K
from datetime import datetime
import os.path
import time
from keras.models import load_model
from keras.models import clone_model
from keras.callbacks import TensorBoard

from CircularBuffer import CircularBuffer
from preprocessing import preprocess

### initializing hyparameters for neural network

PROJECT_DIR = os.getcwd()
#Directory where to write event logs and checkpoint.
TRAIN_DIR = 'tf_train_breakout'
# Path of the restore file
RESTORE_FILE_PATH = PROJECT_DIR+'/'+TRAIN_DIR+'/breakout_model_20181212075635.h5'

# Repeat each action selected by the agent this many times.
# Using a value of 4 results in the agent seeing only every 4th input frame
ACTION_REPEAT = 4
# possible actions: left, right, or stay
ACTION_SIZE = 3
# The number of most recent frames experienced by the agent that are given as input to the Q network""")
AGENT_HISTORY_LENGTH = 4
# dimensions of frames
ATARI_SHAPE = (84, 84, 4)
# size of random batch to train with
BATCH_SIZE = 32
# Discrount factor gamma used in Q-learning update
DISCOUNT_FACTOR = 0.99
# Number of training cases over which each stochastic graduent descent (SGD) update is computed
MINIBATCH_SIZE = 32,
# SGD updates are sampled from this number of most recent frames
REPLAY_MEMORY_SIZE = 400000


GAMMA = 0.99
# Gradient momentum used by RMSProp
GRADIENT_MOMENTUM = 0.95
# The learning rate used b RMSProp
LEARNING_RATE = 0.00025
# Constant added to the squared gradient in the denominator of the RMSProp update
MIN_SQUARED_GRADIENT = 0.01
# Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
NO_OP_MAX = 30
# number of epochs of the optimization loop.""")
NUM_EPISODES = 100000
# Timesteps to observe before training.""")
OBSERVE_STEP_NUM = 1000
# A uniform random policy is run for this number of frames before learning starts and the resulting experience is used to populate the replay memory
REPLAY_START_SIZE = 50000
# Squared gradient (denominator) momentum used by RMSProp
SQUARED_GRADIENT_MOMENTUM = 0.95
# The number of actions selected by the agent between successive SGD updates.
# Using a value of 4 results in the agent selecting 4 actions between each pair of successive updates
UPDATE_FREQUENCY = 4

# Initial value of epsilon in epsilon-greedy exploration
INITIAL_EPSILON = 1
# Final value of epsilon in epsilon-geredy exploration
FINAL_EPSILON = 0.1
# The number of frames over which the initial value of c is linearly annealed to its final value
FINAL_EXPLORATION_FRAME = 1000000
# Linear slope for decrease in epsilon per iteration
EPSILON_SLOPE = (FINAL_EPSILON - INITIAL_EPSILON)/FINAL_EXPLORATION_FRAME
# frames over which to anneal epsilon
REFRESH_TARGET_MODEL_NUM = 10000

# Whether to resume from previous checkpoint
RESUME = False
# Whether to display the game
RENDER = False

def pre_processing(observe):
    processed_observe = np.uint8(
        resize(rgb2gray(observe), (84, 84), mode='constant') * 255)
    return processed_observe

# Transform function from: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
# This will turn the number into -1, 0, 1 which seems to work, but is hacky
def transfrom_reward(reward):
    return np.sign(reward)

"""
Deep Q-Learning Notes

Q(s, a) = r + ( gamma * max_a'(Q(s', a'))

The algorithm plays the game, each step saving
    the initial state,
    the action it took,
    the reward it got,
    and the next state it reached
and this data will be used to train a neural network with Keras

Neural Network
    Input:
        - Initial state (STACK of 4 preprocessed frames)
        - NOTE: We do not pass in an action because we would need to do a foward pass for each action which scales linearly
            - We instead give a separate output for each possible action
            - The outputs are multiplied by a "mask" corresponding to the one-hot encoded action, which will set all its outputs
                to 0 except for the action we actually saw
                - Think of the neurons and probabilities for each

    Output:
        - Estimate of Q(s, a)
            - r is the reward obtained when playing
            - gamma is our discount rate (0.99)
            - Q(s', a') comes from predicting the Q function for the next state using our current model


"""

def fit_batch(model, start_states, actions, rewards, next_states, is_terminal, gamma=0.99):
    """
    Do one deep Q learning iteration.

    model: The DQN (Deep Q-Network)
    gamma (float): discount factor
    start_states (numpy array): array of starting states
    actions (numpy array): array of encoded actions corresponding to the start states
    rewards (numpy array): array of rewards corresponding to the start states and actions
    next_states (numpy array): array of the resulting states corresponding to the start states and actions
    is_terminal (numpy array): boolean array of whether the resulting state is terminal

    """


    # Predicting Q-values of next-states.
    # Masking all states with ones so network considers them as equivalent in priority
    predicted_q_values = model.predict([next_states, np.ones(actions.shape)])

    # By definition the terminal state is 0, so we need to make sure that it is set to 0
    predicted_q_values[is_terminal] = 0

    # Calculate thq Q values by: r + ( gamma * max_a'(Q(s', a'))  (NOTE: We are using the predicted Q values)
    Q_values = rewards + (gamma * np.max(predicted_q_values, axis=1))

    # Fit the calculated Q_values to the keras model
    # NOTE: We are using the actions as a mask so that we 0 out all the actions we are not looking at. This enables us
    #   fit our results to the model
    model.fit( [start_states, actions], actions * Q_values[:, None], nb_epoch=1, batch_size=len(start_states), verbose=0)

# This function is from: https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
# Which is an implementation of model described in the paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
# The neural network takes an input of 84x84x4 and after 4 hidden layers it outputs a fully connected
#   linear layer with a single output for each valid action
def atari_model():
    # With the functional API we need to define the inputs.
    frames_input = layers.Input(ATARI_SHAPE, name='frames')
    actions_input = layers.Input((ACTION_SIZE,), name='action_mask')

    # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
    normalized = layers.Lambda(lambda x: x / 255.0, name='normalization')(frames_input)

    # "The first hidden layer convolves 16 8×8 filters with stride 4 with the input image and applies a rectifier nonlinearity."
    conv_1 = layers.convolutional.Conv2D(
        16, (8, 8), strides=(4, 4), activation='relu'
    )(normalized)
    # "The second hidden layer convolves 32 4×4 filters with stride 2, again followed by a rectifier nonlinearity."
    conv_2 = layers.convolutional.Conv2D(
        32, (4, 4), strides=(2, 2), activation='relu'
    )(conv_1)
    # Flattening the second convolutional layer.
    conv_flattened = layers.core.Flatten()(conv_2)
    # "The final hidden layer is fully-connected and consists of 256 rectifier units."
    hidden = layers.Dense(256, activation='relu')(conv_flattened)
    # "The output layer is a fully-connected linear layer with a single output for each valid action."
    output = layers.Dense(ACTION_SIZE)(hidden)
    # Finally, we multiply the output by the mask!
    filtered_output = layers.Multiply(name='QValue')([output, actions_input])

    model = Model(inputs=[frames_input, actions_input], outputs=filtered_output)
    model.summary()
    optimizer = RMSprop(lr=LEARNING_RATE, rho=0.95, epsilon=0.01)
    # model.compile(optimizer, loss='mse')
    # to changed model weights more slowly, uses MSE for low values and MAE(Mean Absolute Error) for large values
    model.compile(optimizer, loss="mse")
    return model

# epsilon will start at 1,
# and linearly decrease until FINAL_EXPLORATION_FRAME, and will stay at 0.1 after
def get_epsilon_for_iteration(iteration):
    if iteration > FINAL_EXPLORATION_FRAME:
        return FINAL_EPSILON
    else:
        return (iteration * EPSILON_SLOPE)+1

def get_action(history, epsilon, iteration, model):
    if random.random() <= epsilon or iteration <= OBSERVE_STEP_NUM:
        return random.randrange(ACTION_SIZE)
    else:
        q_val = model.predict([history, np.ones(ACTION_SIZE).reshape(1, ACTION_SIZE)])
        # returning index of action with largest q_value
        return np.argmax(q_val[0])

def get_one_hot_encoding(targets, size):
    return np.eye(size)[np.array(targets).reshape(-1)]

def train_from_random_batch(memory, model):
    # random_batch = memory.random_sample(BATCH_SIZE)
    random_batch = random.sample(memory, BATCH_SIZE)
    current_frames = np.zeros((BATCH_SIZE, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
    next_frames = np.zeros((BATCH_SIZE, ATARI_SHAPE[0], ATARI_SHAPE[1], ATARI_SHAPE[2]))
    target = np.zeros((BATCH_SIZE))

    action, reward, isDead = [], [], []
    # state, action, new_frame, reward, is_done

    for i, val in enumerate(random_batch):
        # val = tuple(history, action, reward, next_history, done)
        current_frames[i] = val[0]
        action.append(val[1])
        reward.append(val[2])
        next_frames[i] = val[3]
        isDead.append(val[4])

    mask = np.ones((BATCH_SIZE, ACTION_SIZE))
    next_q_vals = model.predict([next_frames, mask])

    for i in range(BATCH_SIZE):
        if isDead[i]:
            # dead in this "branch", so will exclude from consideration
            target[i] = transfrom_reward(reward[i])
        else:
            target[i] = reward[i] + GAMMA * np.max(next_q_vals[0])

    action_one_hot_2D = get_one_hot_encoding(action, ACTION_SIZE)
    # Filling one-hot-encoding 3d_table with their q_values
    target_one_hot = action_one_hot_2D * target[:, None]

    h = model.fit(
        [current_frames, action_one_hot_2D], target_one_hot, epochs=1, batch_size=BATCH_SIZE, verbose=0)

    return h.history['loss'][0]

def train_model():
    env = gym.make('BreakoutDeterministic-v4')

    # memory = CircularBuffer(REPLAY_MEMORY_SIZE)
    memory = deque(maxlen=REPLAY_MEMORY_SIZE)

    n_episode = 0
    epsilon = INITIAL_EPSILON
    n_global_count = 0

    # TODO: Insert a resume function to continue training a model
    if RESUME:
        model = load_model(RESTORE_FILE_PATH)
        # Assume when we restore the model, the epsilon has already decreased to the final value
        epsilon = FINAL_EPSILON
    else:
        model = atari_model()

    cur_date = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    log_dir = "{}/run-{}-log".format(TRAIN_DIR, cur_date)
    file = tf.summary.FileWriter(log_dir, tf.get_default_graph())

    model_target = clone_model(model)
    model_target.set_weights(model.get_weights())

    while n_episode < NUM_EPISODES:
        # Initialize variables for episode
        done = False
        dead = False

        step, score, start_life = 0, 0, 5
        loss = 0.0
        cur_img = env.reset()

        # Using DeepMind's idea: Do nothing at the start to avoid sub-optimal
        for _ in range(random.randint(1, NO_OP_MAX)):
            cur_img, _, _, _ = env.step(1)

        state = pre_processing(cur_img)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))

        # Playing the whole game (a episode)
        while not done:
            if RENDER:
                env.render()
                time.sleep(0.01) # Need delay for render to load

            # gets action index for the current
            action = get_action(history, epsilon, n_global_count, model_target)

            # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
            state, reward, done, info = env.step(action + 1)

            next_state = pre_processing(state)
            next_state = np.reshape([next_state], (1, 84, 84, 1))

            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            # if the agent missed ball, agent is dead --> episode is not over
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            memory.append((history, action, reward, next_history, done))

            # Start modifying the model when the memory is ready
            if n_global_count > OBSERVE_STEP_NUM:
                # Sample and fit
                loss += train_from_random_batch(memory, model)

                # Update the target model
                if n_global_count % REFRESH_TARGET_MODEL_NUM == 0:
                    model_target.set_weights(model.get_weights())

            score += reward

            # If agent is dead, reset agent to not dead, but keep the history unchanged
            if dead:
                dead = False
            else:
                history = next_history

            n_global_count += 1
            step += 1

            if done:
                if n_global_count <= OBSERVE_STEP_NUM:
                    state = "epsiode"
                elif OBSERVE_STEP_NUM < n_episode <= OBSERVE_STEP_NUM + FINAL_EXPLORATION_FRAME:
                    state = "explore"
                else:
                    state = "train"
                print('state: {}, episode: {}, score: {}, global_step_count: {}, avg loss: {}, step: {}, memory length: {}'
                      .format(state, n_episode, score, n_global_count, loss / float(step), step, len(memory)))

                if n_episode%100 == 0 or (n_episode + 1) == NUM_EPISODES:
                    cur_date = datetime.utcnow().strftime("%Y%m%d%H%M%S")
                    file_name = "breakout_model_{}.h5".format(cur_date)
                    model_path = os.path.join(TRAIN_DIR, file_name)
                    model.save(model_path)

                # Add custom user data to TensorBoard
                loss_summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss / float(step))])
                file.add_summary(loss_summary, global_step=n_global_count)

                score_summary = tf.Summary(value=[tf.Summary.Value(tag="score", simple_value=score)])
                file.add_summary(score_summary, global_step=n_global_count)

                n_episode += 1
    file.close()

def run_model():
    env = gym.make('BreakoutDeterministic-v4')

    n_episode = 0
    epsilon = 0.001
    n_global_count = OBSERVE_STEP_NUM + 1

    model = load_model(RESTORE_FILE_PATH)

    while n_episode < NUM_EPISODES:
        # Initialize variables for episode
        done = False
        dead = False

        score, start_life = 0, 5
        cur_img = env.reset()
        cur_img, _, _, _ = env.step(1)

        state = pre_processing(cur_img)
        history = np.stack((state, state, state, state), axis=2)
        history = np.reshape([history], (1, 84, 84, 4))
        # history = np.reshape([history], 1, 84, 84, 4)

        # Playing the whole game (a episode)
        while not done:
            env.render()
            time.sleep(0.01) # Need delay for render to load

            # gets action index for the current
            action = get_action(history, epsilon, n_global_count, model) + 1

            # Play one game iteration (note: according to the next paper, you should actually play 4 times here)
            new_frame, reward, is_done, info = env.step(action)

            next_state = pre_processing(new_frame)
            next_state = np.reshape([next_state], (1, 84, 84, 1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            # if the agent missed ball, agent is dead --> episode is not over
            if start_life > info['ale.lives']:
                dead = True
                start_life = info['ale.lives']

            reward = np.clip(reward, -1., 1)
            score += reward

            # If agent is dead, reset agent to not dead, but keep the history unchanged
            if dead:
                dead = False
            else:
                history = next_history

            n_global_count += 1

            if done:
                n_episode += 1
                print('episode: {}, score{}').format(n_episode, score)
    file.close()
if __name__ == "__main__":
    # print("test")
    train_model()
    # run_model()