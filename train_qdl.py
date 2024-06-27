from model import LSTM_Trader
import torch
from collections import deque
import random
from dataloader import df_bit

# ┬ ┬┬┌─┐┌─┐┬─┐┌─┐┌─┐┬─┐┌─┐┌┬┐┌─┐┌┬┐┌─┐┬─┐┌─┐
# ├─┤│├─┘├┤ ├┬┘├─┘├─┤├┬┘├─┤│││├┤  │ ├┤ ├┬┘└─┐
# ┴ ┴┴┴  └─┘┴└─┴  ┴ ┴┴└─┴ ┴┴ ┴└─┘ ┴ └─┘┴└─└─┘
#######################################################
# H = 0.01
LR = 1e-3
epsilon = 0.1

#######################################################

class ReplayMemory:

    def __init__(self, maxlen):
        self.memory = deque([], maxlen=maxlen)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, sample_size):
        return random.sample(self.memory, sample_size)

    def __len__(self):
        return len(self.memory)



#######################################################

# make a nn called DQN which takes as input:
# - hidden state
# - current price
# - if we own btc or not
#
# and returns long time reward for :
# - buy all in
# - sell all
# - hold

class Trainer:

    def __init__(self):
        """
        Networks are generated here
        Load the dataset here
        Replay memory is generated here (TODO)
        Logging is generated here (TODO)
        """
        self.state_size = 20
        input_size = 2 # current price and a boolean indicating if the agent owns btc or not
        output_size = 3 # buy, sell, hold
        #memory_size = 100

        #memory = ReplayMemory(memory_size)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.policy_net = LSTM_Trader(self.state_size, input_size, output_size).to(self.device)
        self.target_net = LSTM_Trader(self.state_size, input_size, output_size).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.optimizer_target = torch.optim.Adam(self.target_net.parameters(), lr=LR)

        self.train_dataset = df_bit.iloc[:int(0.8*len(df_bit))]
        self.test_dataset = df_bit.iloc[int(0.8*len(df_bit)):]


    def train(self):
        """
        Train the model
        """
        n_epochs = 100
        train_size = 1_000
        # Train the model
        for epoch in range(n_epochs):
            # Extract a random segment of the training dataset
            random_idx = random.randint(0, len(self.train_dataset) - train_size)
            train_segment = self.train_dataset.iloc[random_idx:random_idx + train_size]
            print(f'Epoch {epoch}')
            # Generate a neutral hidden state and cell state
            h = torch.ones(self.state_size)
            c = torch.ones(self.state_size)
            h = h.to(torch.device(self.device))
            c = c.to(torch.device(self.device))
            total_loss = 0
            own_btc = 0
            for i in range(len(train_segment) - 1):
                x = train_segment['price'].iloc[i]
                label = train_segment['delta_price'].iloc[i]
                # Convert to tensor and send to GPU
                # Make x a tensor of shape (1)
                x = torch.ones(self.state_size) * x
                label = torch.tensor([label]).float()
                current_price = x.to(torch.device(self.device))
                label = label.to(torch.device(self.device))
                # Forward pass
                with torch.no_grad():
                    h_target, c_target = self.policy_net(h, c, current_price, own_btc)

                action = self.greedy_choice(h_target)

                reward = self.calculate_reward_d(current_price, action, own_btc)

                # Update the state
                own_btc =

    # choose an action A* using epsilon greedy policy (hyperparameter epsilon)
    # - with probability epsilon choose a random action
    # - with probability 1-epsilon choose the action with the highest Q value calculated by the policy_net

    def greedy_choice(self, h_target):
        if torch.rand(1) < epsilon:
            return torch.randint(0, 3, (1,))
        else:
            return h_target.argmax(0)


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
        dumb_penalty = 0.01
        if (last_action == 0):
            # buy, if you already own btc, then you get a penalty
            if own_btc:
                return -dumb_penalty
            return torch.log(current_price/last_price)
        if (last_action == 1):
            # sell, if you do not own btc, then you get a penalty
            if not own_btc:
                return -dumb_penalty
            return -torch.log(current_price/last_price)
        if (last_action == 2):
            if own_btc:
            # hold
                return torch.log(current_price/last_price)
            else:
                return -torch.log(current_price/last_price)







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