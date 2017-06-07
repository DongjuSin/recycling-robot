import pygame
import sys
import tensorflow as tf
import numpy as np
import random
import os
import math
from Simulator import World, Thing, Robot, Can

# Parameters
epsilon = 1 # The probability of choosing a random action (in training).
min_epsilon = 0.001 # The minimum value we want epsilon to reach in training.
epoch = 1001 # The number of games we want the system to run for.
state_size = 32
n_state = state_size*state_size
hidden_size = state_size*state_size
n_action = 8 # 9? wheel_move?
learning_rate = 0.2
maxMemory = 500 # How large should the memory be(where it stores its past experiences).
discount = 0.9 # The discount is used to force the network to choose states that lead to the reward quicker
learning_rate = 0.2
batchSize = 50

# Create the base model.
X = tf.placeholder(tf.float32, [None, n_state])
W1 = tf.Variable(tf.truncated_normal([n_state, hidden_size], stddev=1.0/math.sqrt(float(n_state))))
b1 = tf.Variable(tf.truncated_normal([hidden_size], stddev=0.01))
input_layer = tf.nn.relu(tf.matmul(X, W1) + b1)
W2 = tf.Variable(tf.truncated_normal([hidden_size, hidden_size], stddev=1.0/math.sqrt(float(hidden_size))))
b2 = tf.Variable(tf.truncated_normal([hidden_size], stddev=0.01))
hidden_layer = tf.nn.relu(tf.matmul(input_layer, W2) + b2)
W3 = tf.Variable(tf.truncated_normal([hidden_size, n_action], stddev=1.0/math.sqrt(float(hidden_size))))
b3 = tf.Variable(tf.truncated_normal([n_action], stddev=0.01))
output_layer = tf.matmul(hidden_layer, W3) + b3
# True labels
Y = tf.placeholder(tf.float32, [None, n_action])
# Mean squared error cost function
cost = tf.reduce_sum(tf.square(Y-output_layer)) # / (2*batch_size)
# Stochastic Gradient Descent Optimizer
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Helper function: Choose a random value between the two boundaries.
def randf(s,e):
    return (float(random.randrange(0, (e-s)*9999)) / 10000) + s;

world = None
ACTIONS= {
    "move_forward": lambda world, v: world.robot.move_forward(v),
    "move_backward": lambda world, v: world.robot.move_backward(v),
    "move_left": lambda world, v: world.robot.move_right(v),
    "move_right": lambda world, v: world.robot.move_left(v),
    "turn_left": lambda world, w: world.robot.rotate_left(w),
    "turn_right": lambda world, w: world.robot.rotate_right(w),
    "grab": lambda world: world.robot.try_grab_nearlist(world),
    "put": lambda world: world.robot.put(),
    "wheel_move": lambda world, lt, rt, lb, rb: world.robot.wheel_move(lt, rt, lb, rb)
}

def gen_world(width, height):
    world = World(width, height)
    return world

def init_world(world):
    world.__init__(world.w, world.h)

def act(world, action):
    if(action == 1):
        send_action(world, "move_forward", 1)
    elif(action == 2):
        send_action(world, "turn_left", 1)
    elif(action == 3):
        send_action(world, "turn_right", 1)
    elif(action == 4):
        send_action(world, "move_backward", 1)
    elif(action == 5):
        send_action(world, "grab")
    elif(action == 6):
        send_action(world, "put")
    elif(action == 7):
        send_action(world, "move_left", 1)
    elif(action == 8):
        send_action(world, "move_right", 1)


def send_action(world, action, *args, **kwargs):
    ACTIONS[action](world, *args, **kwargs)

def get_reward(world):
    return world.get_score()

def get_screen_pixels(world, w, h):
    #return pygame.surfarray.array2d(pygame.transform.scale(world.get_screen(), (w, h))).view('uint8').reshape((w, h, 4,))[..., :3][:,:,::-1]

    return pygame.surfarray.array2d(pygame.transform.scale(world.get_screen(), (w, h))).view('uint8').reshape((w, h, 4,))[..., :3][:,:,::-1]

def rgb2gray(rgb):
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989*r + 0.5870*g + 0.1140*b

    return gray

def observe(world, w, h):
    state = get_screen_pixels(world, w, h)
    state = rgb2gray(state)
    state = np.reshape(state, (1, n_state))
    return state

clock = pygame.time.Clock()
TITLE="Recycler"
pygame.init()
last_w=0
last_h=0
screen = pygame.display.set_mode((last_w, last_h))
def show(world):
    if last_w!=world.w or last_h!=world.h:
        screen = pygame.display.set_mode((world.w, world.h))
    world.draw_on(screen)
    pygame.display.flip()
    clock.tick()
    pygame.display.set_caption(TITLE + "/FPS: "+ str(round(clock.get_fps())) + "/SCORE: " + str(get_reward(world)))


# The memory : Handles the internal memory that we add experiences that occur based on agent's actions,
# and creates batches of experiences based on the mini-batch size for training.
class ReplayMemory:
    def __init__(self, state_size, maxMemory, discount):
        self.maxMemory = maxMemory
        self.state_size = state_size
        self.n_state = state_size*state_size
        self.discount = discount

        canvas = np.zeros((self.state_size, self.state_size))
        canvas = np.reshape(canvas, (-1,self.n_state))
        # self.inputState = np.empty((self.maxMemory, 100), dtype = np.float32)
        self.inputState = np.empty((self.maxMemory, 32*32), dtype = np.uint8)
        self.actions = np.zeros(self.maxMemory, dtype = np.uint8)
        # self.nextState = np.empty((self.maxMemory, 100), dtype = np.float32)
        self.nextState = np.empty((self.maxMemory, 32*32), dtype = np.uint8)
        self.gameOver = np.empty(self.maxMemory, dtype = np.bool)
        self.rewards = np.empty(self.maxMemory, dtype = np.int8)
        self.count = 0
        self.current = 0

    # Appends the experience to the memory
    def remember(self, currentState, action, reward, nextState, gameOver):
        self.actions[self.current] = action
        self.rewards[self.current] = reward
        # print(self.inputState[self.current].shape, currentState[0].shape)
        # print(self.inputState[self.current].dtype, currentState[0].dtype)
        # print(self.inputState[self.current, 0],currentState[0,0])
        self.inputState[self.current, ...] = currentState[0]
        self.nextState[self.current, ...] = nextState[0]
        self.gameOver[self.current] = gameOver
        self.count = max(self.count, self.current + 1)
        self.current = (self.current + 1) % self.maxMemory 

    def getBatch(self, model, batchSize, nbActions, nbStates, sess, X):
    
        # We check to see if we have enough memory inputs to make an entire batch, if not we create the biggest
        # batch we can (at the beginning of training we will not have enough experience to fill a batch).
        memoryLength = self.count
        chosenBatchSize = min(batchSize, memoryLength)

        inputs = np.zeros((chosenBatchSize, nbStates))
        targets = np.zeros((chosenBatchSize, nbActions))

        # Fill the inputs and targets up.
        for i in range(chosenBatchSize):
            if memoryLength == 1:
                memoryLength = 2
            # Choose a random memory experience to add to the batch.
            randomIndex = random.randrange(1, memoryLength)
            #current_inputState = np.reshape(self.inputState[randomIndex], (1, 100))
            current_inputState = np.reshape(self.inputState[randomIndex], (1,32*32))

            target = sess.run(model, feed_dict={X: current_inputState})
      
            #current_nextState =  np.reshape(self.nextState[randomIndex], (1, 100))
            current_nextState = np.reshape(self.nextState[randomIndex], (1,32*32))
            current_outputs = sess.run(model, feed_dict={X: current_nextState})      
      
            # Gives us Q_sa, the max q for the next state.
            nextStateMaxQ = np.amax(current_outputs)
            if (self.gameOver[randomIndex] == True):
                target[0, [self.actions[randomIndex]-1]] = self.rewards[randomIndex]
            else:
                # reward + discount(gamma) * max_a' Q(s',a')
                # We are setting the Q-value for the action to  r + gamma*max a' Q(s', a'). The rest stay the same
                # to give an error of 0 for those outputs.
                target[0, [self.actions[randomIndex]-1]] = self.rewards[randomIndex] + self.discount * nextStateMaxQ

            # Update the inputs and targets.
            inputs[i] = current_inputState
            targets[i] = target

        return inputs, targets


def main(_):
    world = gen_world(800,800)
    show(world)

    # have to initialize replay memory
    memory = ReplayMemory(state_size, maxMemory, discount)
    # havae to initialize action-value function Q with random weights

    # Observe initial state s


    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()

        for i in range(epoch):
            err = 0
            isgameover = False
            #current_state = get_screen_pixels(world, 32, 32)
            current_state = observe(world, 32, 32)

            while(isgameover != True): #condition
                #initialize clock
                world.set_t_start()
                # initialize action
                action = -9999
                # Decides if we should choose a random action, or an action from the policy network.
                global epsilon
                if(randf(0,1) <= epsilon):
                    action = random.randrange(1, n_action+1)
                    print("action is : " + str(action))
                else:
                    # Forward the current state through the network.
                    q = sess.run(output_layer, feed_dict={X: current_state})
                    # Find the max index
                    index = q.argmax()
                    action = index + 1
                
                # Decay the epsilon by multiplying by 0.999, not allowing it to go below a certain threshold.
                if(epsilon > min_epsilon):
                    epsilon = epsilon * 0.999

                # carry out an action
                act(world,action)
                print("here1")
                # get next state and reward
                #next_state = get_screen_pixels(world, 32, 32)
                next_state = observe(world, 32, 32)
                reward = world.get_score()
                isgameover = world.is_gameover()
                print("here2")
                # store experience <s,a,r,s'> in replay memory
                memory.remember(current_state, action, reward, next_state, isgameover)
                # update current state and if the game is over
                current_state = next_state
                
                # get a batch of training data to train the model
                inputs, targets = memory.getBatch(output_layer, batchSize, n_action, n_state, sess, X)

                # Train the network which returns the error
                _, loss = sess.run([train_step, cost], feed_dict={X: inputs, Y: targets})
                err = err + loss
            print("Epoch " + str(i) + ": err = " + str(err))
        # Save the variables to disk.
        save_path = saver.save(sess, os.getcwd()+"/model.ckpt")
        print("Model saved in file: %s" % save_path)

def test(world, w=0, h=0):
    '''
    This function is just for the test.
    You can see the gui and you can test the action works well by a keyboard.
    '''
    while True:
        show(world)
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            send_action(world, "move_forward", 1)
        if keys[pygame.K_a]:
            send_action(world, "turn_left", 1)
        if keys[pygame.K_d]:
            send_action(world, "turn_right", 1)
        if keys[pygame.K_s]:
            send_action(world, "move_backward", 1)
        if keys[pygame.K_UP]:
            send_action(world, "grab")
        if keys[pygame.K_DOWN]:
            send_action(world, "put")
        if keys[pygame.K_q]:
            send_action(world, "move_left", 1)
            #send_action(world, "wheel_move", 1, -1, -1, 1)
        if keys[pygame.K_e]:
            send_action(world, "move_right", 1)
            #send_action(world, "wheel_move", -1, 1, 1, -1)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                pass
            elif event.type == pygame.MOUSEBUTTONUP:
                pass
            elif event.type == pygame.MOUSEMOTION:
                pass


#world = gen_world(800, 800)
#test(world, 800, 800)
if __name__ == '__main__':
    tf.app.run()

