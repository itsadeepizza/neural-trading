from model import LSTM_Trader
import torch


# ┬ ┬┬┌─┐┌─┐┬─┐┌─┐┌─┐┬─┐┌─┐┌┬┐┌─┐┌┬┐┌─┐┬─┐┌─┐
# ├─┤│├─┘├┤ ├┬┘├─┘├─┤├┬┘├─┤│││├┤  │ ├┤ ├┬┘└─┐
# ┴ ┴┴┴  └─┘┴└─┴  ┴ ┴┴└─┴ ┴┴ ┴└─┘ ┴ └─┘┴└─└─┘
#######################################################
# H = 0.01
LR = 1e-3
epsilon = 0.1


#######################################################

# make a nn called DQN which takes as input:
# - hidden state
# - current price
# - invested amount (in btc)
# - available amount (in dollars)
# - (optional) last available amount (in dollars)
#
# and returns long time reward for :
# - buy all in
# - sell all
# - hold

state_size = 20
input_size = 2 # current price and a boolean indicating if the agent owns btc or not
output_size = 3 # buy, sell, hold
policy_net = LSTM_Trader(state_size, input_size, output_size)
target_net = LSTM_Trader(state_size, input_size, output_size)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=LR)
optimizer_target = torch.optim.Adam(target_net.parameters(), lr=LR)
policy_net.to(device)
target_net.to(device)


# Make a function calculate_reward which takes as input:
# - available amount (in dollars)
# - last available amount (in dollars)
# - last action
#
# and returns the reward for the last action as a float
#  The reward is calculated as follows:
# A)
# - if the last action was buy the reward is 0
# - if the last action was sell the reward is the difference between the current available amount and the last available amount
# - if the last action was hold the reward is -H (where H is a hyperparameter)
#
# B) reward could be the ratio between the current available amount and the last available amount (bad idea because
# of how long term reward is calculated)
#
# C)
# - if the last action was buy the reward is 0
# - if the last action was hold, but you do not own any btc the reward is -H
# - if the last action was hold, but you own btc the reward is the difference between the current estimated value of the btc and the last estimated value of the btc (in dollars)
# - if the last action was sell the reward is the same as above
#
# D)
# current price = p_t, last price = p_t-1
# - if the last action was buy the reward is log(p_t/p_t-1)
# - if the last action was hold, but you do not own any btc the reward is -log(p_t/p_t-1)
# - if the last action was hold, but you own btc the reward is log(p_t/p_t-1)
# - if the last action was sell the reward is -log(p_t/p_t-1)
# This sums up pretty well, as the sum of the rewards is the log of the ratio of the final price to the initial price
# log(p_t/p_0) = log(p_1/p_0) + log(p_2/p_1) + ... + log(p_t/p_t-1)
#
def calculate_reward_a(available_amount, last_available_amount, last_action):
    # buy 0 sell 1 hold 2
    if(last_action == 0):
        return 0
    if(last_action == 1):
        return available_amount-last_available_amount
    if(last_action == 2):
        return -H


def calculate_reward_b(available_amount, last_available_amount, last_action):
    # buy 0 sell 1 hold 2
    return available_amount/last_available_amount


def calculate_reward_c(available_amount, last_available_amount, last_action, estimate, last_estimate):
    # buy 0 sell 1 hold 2
    if(last_action == 0):
        return 0
    if(last_action == 1):
        return available_amount-last_available_amount
    if(last_action == 2):
        if(available_amount == 0):
            return -H
        elif(available_amount > 0):
            return estimate - last_estimate #estimated????
        else:
            raise ValueError

def calculate_reward_d(current_price, last_price, last_action, own_btc):
    # buy 0 sell 1 hold 2
    if (last_action == 0):
        # buy, if you already own btc, is the same as hold
        return torch.log(current_price/last_price)
    if (last_action == 1):
        # sell, if you do not own btc, is the same as hold
        return -torch.log(current_price/last_price)
    if (last_action == 2):
        if own_btc:
        # hold
            return torch.log(current_price/last_price)
        else:
            return -torch.log(current_price/last_price)





# choose an action A* using epsilon greedy policy (hyperparameter epsilon)
# - with probability epsilon choose a random action
# - with probability 1-epsilon choose the action with the highest Q value calculated by the policy_net

def greedy_choice(last_price, own_btc):
    if torch.rand(1) < epsilon:
        return torch.randint(0, 3, (1,))
    else:
        return policy_net(last_price, own_btc).argmax(0)


# calculate the reward for the last action

# ???? store the transition in the replay memory

def update_state(action):
    # update the state with the new action
    if action == 0:
        own_btc = True
    elif action == 1:
        own_btc = False

# The action is executed and the new state is observed

# policy_net(state)(A*) = Q(state, A*)
# max(target_net(new state)) = mmax_i(Q(new state, Ai))
#
# Q(state, A*) = reward + gamma * max_i(Q(new state, Ai))
# delta = Q(state, A*) - (reward + gamma * max_i(Q(new state, Ai))) = loss

# update the weights of the policy_net using the loss

# every C steps update the target_net with the weights of the policy_net