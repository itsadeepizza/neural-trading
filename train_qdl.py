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

# policy_net = DQN().to(device)
# target_net = DQN().to(device)


# choose an action A* using epsilon greedy policy (hyperparameter epsilon)
# - with probability epsilon choose a random action
# - with probability 1-epsilon choose the action with the highest Q value calculated by the policy_net


# calculate the reward for the last action

# ???? store the transition in the replay memory


# The action is executed and the new state is observed

# policy_net(state)(A*) = Q(state, A*)
# max(target_net(new state)) = mmax_i(Q(new state, Ai))
#
# Q(state, A*) = reward + gamma * max_i(Q(new state, Ai))
# delta = Q(state, A*) - (reward + gamma * max_i(Q(new state, Ai))) = loss

# update the weights of the policy_net using the loss

# every C steps update the target_net with the weights of the policy_net