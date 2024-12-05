from model import LSTM_Trader
import torch
from collections import deque
import random
from merge_dataset import load_dataset
import math
import os

USE_WANDB = False
USE_REPLAY_MEMORY = True

if USE_WANDB:
    import wandb



# ┬ ┬┬┌─┐┌─┐┬─┐┌─┐┌─┐┬─┐┌─┐┌┬┐┌─┐┌┬┐┌─┐┬─┐┌─┐
# ├─┤│├─┘├┤ ├┬┘├─┘├─┤├┬┘├─┤│││├┤  │ ├┤ ├┬┘└─┐
# ┴ ┴┴┴  └─┘┴└─┴  ┴ ┴┴└─┴ ┴┴ ┴└─┘ ┴ └─┘┴└─└─┘
#######################################################
# H = 0.01
LR = 1e-5
epsilon_start = 0.6
epsilon_end = 0.01
time_interval = 500 # epochs
gamma = 0.99

#----------------------------------------------------------

# calculate epsilon decay
epsilon_decay = (epsilon_end / epsilon_start) ** (1 / time_interval)
# well done copilot !
epsilon = epsilon_start

#######################################################


# ╦  ╔═╗╔═╗╔═╗╦╔╗╔╔═╗         ╦ ╦╔═╗╔╗╔╔╦╗╔╗
# ║  ║ ║║ ╦║ ╦║║║║║ ╦  ───    ║║║╠═╣║║║ ║║╠╩╗
# ╩═╝╚═╝╚═╝╚═╝╩╝╚╝╚═╝         ╚╩╝╩ ╩╝╚╝═╩╝╚═╝

if USE_WANDB:
    # Start a new wandb run to track this train
    wandb.init(
        # set the wandb project where this run will be logged
        project="Neural_Trading",
    
        # track hyperparameters and run metadata
        config={
        "learning_rate": LR,
        "epsilon_start": epsilon_start,
        "epsilon_end": epsilon_end,
        "time_interval": time_interval,
        "gamma": gamma,
        }
    )


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
        self.input_size = 2
        output_size = 2 # buy, sell

        self.memory = ReplayMemory(1000)

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.policy_net = LSTM_Trader(self.state_size, self.input_size, output_size).to(self.device)
        self.target_net = LSTM_Trader(self.state_size, self.input_size, output_size).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer_policy = torch.optim.Adam(self.policy_net.parameters(), lr=LR)
        self.optimizer_target = torch.optim.Adam(self.target_net.parameters(), lr=LR)

        # Load the datasets
        self.train_dataset = load_dataset('dataset/train')
        self.test_dataset = load_dataset('dataset/test')


        # initialize chunks for test dataset
        test_size = 1_000
        num_chunks = 10
        test_chunks = [random.randint(0, len(self.test_dataset) - test_size) for j in range(num_chunks)]
        self.test_segments = [self.test_dataset.iloc[chunk:chunk + test_size] for chunk in test_chunks]

        self.test_capital = 1
        self.test_capital_always_buy = 1
        self.test_capital_crazy_monkey = 1

        self.test_capital_crazy_monkey_B = 1

        self.test_capital_savant_monkey = 1


    def test(self):

        # Generate starting h and C
        h = torch.ones(self.state_size).to(torch.device(self.device))
        c = torch.ones(self.state_size).to(torch.device(self.device))

        last_delta = 0
        savant_monkey_prob = 0.5
        
        for test_segment in self.test_segments:
            own_btc = 0
            for i in range(len(test_segment) - 1):

                raw_current_price = test_segment['price'].iloc[i]
                raw_new_price = test_segment['price'].iloc[i + 1]

                # Normalize the price
                current_price = (raw_current_price - 65481) / 25580
                new_price = (raw_new_price - 65481) / 25580

                x = torch.tensor([current_price, own_btc], dtype = torch.float32).to(torch.device(self.device))

                # Forward pass
                with torch.no_grad():
                    # Calculate max_a Q(state, a)
                    policy_out, h, c = self.policy_net(h, c, x)
                    action = policy_out.argmax(0)


                # UPDATE THE STATE AND CAPITAL

                # Update dummy capitals
                self.test_capital_always_buy = (raw_new_price / raw_current_price) * self.test_capital_always_buy
                if torch.rand(1) < 0.5:
                    self.test_capital_crazy_monkey = (raw_new_price / raw_current_price) * self.test_capital_crazy_monkey

                if torch.rand(1) < 0.5:
                    self.test_capital_crazy_monkey_B = (raw_new_price / raw_current_price) * self.test_capital_crazy_monkey_B
                    
                savant_monkey_prob += last_delta * 100 * 0.02
                savant_monkey_prob = min(1, max(0, savant_monkey_prob))
                if torch.rand(1) < savant_monkey_prob:
                    self.test_capital_savant_monkey = (raw_new_price / raw_current_price) * self.test_capital_savant_monkey

                if USE_WANDB:
                    wandb.log({"savant_monkey_prob": savant_monkey_prob, 
                               "test_capital_savant_monkey": self.test_capital_savant_monkey,
                                "test_capital_crazy_monkey_B": self.test_capital_crazy_monkey_B,
                                "test_capital_crazy_monkey": self.test_capital_crazy_monkey,
                                "test_capital_always_buy": self.test_capital_always_buy,
                                "test_capital": self.test_capital,
                                "test_current_price": x[0]
                               })


                # update own_btc
                if action == 0: # buy
                    own_btc = 1
                    self.test_capital = (raw_new_price / raw_current_price) * self.test_capital
                elif action == 1: # sell
                    own_btc = 0
                else:
                    raise RuntimeError('Impossible action')
                
                last_delta = new_price - current_price



    def train_with_rm(self):
        """
        train using episodes, an episode starts with a random segment of 
        the dataset and goes on until the agent buy and sell (therefore
        observing a reward) or the segment ends
        """

        global epsilon
        n_episodes = 1000
        train_size = 10_000

        # fill memory with transitions while interacting with the environment

        for episode in range(n_episodes):

            terminated = False # after a buy and a sell
            truncated = False # after the current segment ends

            random_idx = random.randint(0, len(self.train_dataset) - train_size)
            train_segment = self.train_dataset.iloc[random_idx:random_idx + train_size]

            while (not terminated and not truncated):

                # Generate starting h and C
                h = torch.ones(self.state_size).to(torch.device(self.device))
                c = torch.ones(self.state_size).to(torch.device(self.device))

                own_btc = 0 # 0 = No btc, 1 = Own btc
                new_own_btc = own_btc

                for i in range(len(train_segment) - 1):

                    raw_current_price = train_segment['price'].iloc[i]
                    raw_new_price = train_segment['price'].iloc[i + 1]

                    # Normalize the price
                    current_price = (raw_current_price - 65481) / 25580
                    new_price = (raw_new_price - 65481) / 25580

                    x = torch.tensor([current_price, own_btc], dtype = torch.float32).to(torch.device(self.device))

                    # Get the action for current environment state
                    with torch.no_grad():
                        policy_out, new_h, new_c = self.policy_net(h, c, x)

                    action = self.epsilon_greedy_choice(policy_out) # 0 = buy, 1 = sell

                    if action == 0 and own_btc == 0:
                        new_own_btc == 1
                    if action == 1 and own_btc == 1:
                        terminated = True

                    new_x = torch.tensor([new_price, new_own_btc], dtype = torch.float32).to(torch.device(self.device))

                    # Compute reward for chosen action
                    reward = self.calculate_reward_e(new_price, current_price, action, own_btc)

                    env_state = (h, c, x)
                    new_env_state = (new_h, new_c, new_x)

                    self.memory.append((env_state, action, new_env_state, reward, terminated))

                    h = new_h
                    c = new_c
                    own_btc = new_own_btc

                truncated = True

            # sample transitions from the memory and use them to update the weights

            memory_batch = self.memory.sample(10)

            for (env_state, action, new_env_state, reward, terminated) in memory_batch:

                # get the target return for the action used in transition
                if terminated:
                    target_return = reward 
                else:
                    with torch.no_grad():
                        target_return = reward + gamma * self.target_net(new_env_state[1],new_env_state[2],new_env_state[0])[0].max()
            
                with torch.no_grad():
                    tar_Q_values = self.target_net(env_state[0],env_state[1],env_state[2])[0]        

                tar_Q_values[action] = target_return 
                pol_Q_values = self.policy_net(env_state[0],env_state[1],env_state[2])[0] # gradient is computed here
            
                loss = torch.nn.MSELoss()(pol_Q_values, tar_Q_values)
                print(loss.item())

                # Backward pass
                self.optimizer_policy.zero_grad()
                loss.backward()
                # Update the weights
                self.optimizer_policy.step()
            
            
                


    def train(self):
        """
        Train the model
        """
        global epsilon
        n_epochs = 1000
        train_size = 10_000

        print()

        capital = 1
        capital_always_buy = 1
        capital_crazy_monkey = 1
        # Train the model
        for epoch in range(n_epochs):

            # Extract a random segment of the training dataset
            random_idx = random.randint(0, len(self.train_dataset) - train_size)
            train_segment = self.train_dataset.iloc[random_idx:random_idx + train_size]

            print(f'Epoch {epoch}')

            # Generate starting h and C
            h = torch.ones(self.state_size).to(torch.device(self.device))
            c = torch.ones(self.state_size).to(torch.device(self.device))
            h_target = torch.ones(self.state_size).to(torch.device(self.device))
            c_target = torch.ones(self.state_size).to(torch.device(self.device))

            total_loss = 0
            own_btc = 0 # 0 = No btc, 1 = Own btc

            for i in range(len(train_segment) - 1):

                raw_current_price = train_segment['price'].iloc[i]
                raw_new_price = train_segment['price'].iloc[i + 1]

                # Normalize the price
                current_price = (raw_current_price - 65481) / 25580
                new_price = (raw_new_price - 65481) / 25580

                x = torch.tensor([current_price, own_btc], dtype = torch.float32).to(torch.device(self.device))

                # Forward pass
                with torch.no_grad():
                    policy_out, _, _ = self.policy_net(h, c, x)

                action = self.epsilon_greedy_choice(policy_out) # 0 = buy, 1 = sell, 2 = hold

                # if action == 0 and own_btc == 0:
                #     buy_price = current_price
                # if action == 1 and own_btc == 1:
                #     buy_price = None
                
                #reward = self.calculate_reward_d(new_price, current_price, action, own_btc)
                reward = self.calculate_reward_e(new_price, current_price, action, own_btc)

                # Calculate max_a Q(state, a)
                policy_out, h, c = self.policy_net(h, c, x)
                state_action_value = policy_out[action].unsqueeze(0)

                # UPDATE THE STATE
                # print the state of own_btc before updating
                if i % 300 == 0:
                    print(f'own_btc before {own_btc}')
                # update own_btc
                if action == 0:
                    own_btc = 1
                    capital = (raw_new_price / raw_current_price) * capital
                elif action == 1:
                    own_btc = 0
                else:
                    raise RuntimeError('Impossible action')

                # Update dummy capitals
                capital_always_buy = (raw_new_price / raw_current_price) * capital_always_buy
                if torch.rand(1) < 0.5:
                    capital_crazy_monkey = (raw_new_price / raw_current_price) * capital_crazy_monkey

                # update x -> x_new
                x_new = torch.ones(self.input_size - 1, device=self.device) * new_price
                x_new = torch.cat((x_new, torch.tensor([own_btc], device=self.device).float()), 0)

                with torch.no_grad():
                    # Calculate max_i Q(new state, Ai)
                    target_out, h_target, c_target = self.target_net(h_target, c_target, x_new)
                    next_state_value = target_out.max(0)[0]
                    expected_state_action_values = reward + gamma * next_state_value
                    # Change shape to (1)
                    expected_state_action_values = expected_state_action_values.view(1)
                loss = torch.nn.MSELoss()(state_action_value, expected_state_action_values)
                total_loss += loss.item()
                # Backward pass
                self.optimizer_policy.zero_grad()
                loss.backward()
                # Update the weights
                self.optimizer_policy.step()

                # Reset the gradients for h and c
                h = h.detach()
                c = c.detach()
                if i % 300 == 0:
                    print(f'action chosen: {action}')
                    print(f'current price {x[0]} \npolicy output {policy_out} \ntarget net {target_out}\nloss {loss.item()}')
                    print(f'capital {capital}')
                    print('------------------------')
                    # the same as above, but using WandB
                    if USE_WANDB:
                        wandb.log({
                            "action": action,
                            "current_price": x[0],
                            "loss": loss.item(),
                            "capital": capital,
                            "own_btc": own_btc,
                            "epoch": epoch,
                            "capital_always_buy": capital_always_buy,
                            "capital_crazy_monkey": capital_crazy_monkey,
                            "state_action_value": state_action_value,
                            "expected_state_action_values": expected_state_action_values,
                            "epsilon": epsilon,
                            })


                        # log policy output as two plots on the same panel
                        wandb.log({"policy_buy": policy_out[0], "policy_sell": policy_out[1]})

                if i % 5 == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())

            # One epoch finished, a new one will start...
            epsilon = epsilon * epsilon_decay
            print(f'Epoch {epoch} finished, Loss {total_loss / (len(train_segment) - 1)}')

            if epoch % 10 == 0:
                self.test()
                if not os.path.exists('./models'):
                    try:
                        os.mkdir('./models')
                        print("models directory created")
                    except Exception as e:
                        print(f"can't make models directory: {e}")
                        quit()
                print(f'Saving model at epoch {epoch}')
                torch.save(self.policy_net.state_dict(), f'./models/policy_net_{epoch}.pt')
            print()






    # The action is executed and the new state is observed

    # policy_net(state)(A*) = Q(state, A*)
    # max(target_net(new state)) = mmax_i(Q(new state, Ai))
    #
    # Q(state, A*) = reward + gamma * max_i(Q(new state, Ai))
    # delta = Q(state, A*) - (reward + gamma * max_i(Q(new state, Ai))) = loss

    #  Q(state, A*) is calculated by the policy_net and Q(new state, Ai) is calculated by the target_net

    # update the weights of the policy_net using the loss

    # every C steps update the target_net with the weights of the policy_net

    # choose an action A* using epsilon greedy policy (hyperparameter epsilon)
    # - with probability epsilon choose a random action
    # - with probability 1-epsilon choose the action with the highest Q value calculated by the policy_net

    def epsilon_greedy_choice(self, h_target):
        if torch.rand(1) < epsilon:
            return torch.randint(0, 1, (1,)).item()
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
    # def calculate_reward_a(available_amount, last_available_amount, last_action):
    #     # buy 0 sell 1 hold 2
    #     if(last_action == 0):
    #         return 0
    #     if(last_action == 1):
    #         return available_amount-last_available_amount
    #     if(last_action == 2):
    #         return -H
    #
    #
    # def calculate_reward_b(available_amount, last_available_amount, last_action):
    #     # buy 0 sell 1 hold 2
    #     return available_amount/last_available_amount
    #
    #
    # def calculate_reward_c(available_amount, last_available_amount, last_action, estimate, last_estimate):
    #     # buy 0 sell 1 hold 2
    #     if(last_action == 0):
    #         return 0
    #     if(last_action == 1):
    #         return available_amount-last_available_amount
    #     if(last_action == 2):
    #         if(available_amount == 0):
    #             return -H
    #         elif(available_amount > 0):
    #             return estimate - last_estimate #estimated????
    #         else:
    #             raise ValueError

    def calculate_reward_d(self, new_price, last_price, last_action, own_btc):
        # buy 0 sell 1 hold 2
        dumb_penalty = 0.01
        if (last_action == 0):
            # buy, if you already own btc, then you get a penalty
            if own_btc:
                r = -dumb_penalty
            else:
                r = math.log(new_price/last_price)
        if (last_action == 1):
            # sell, if you do not own btc, then you get a penalty
            if not own_btc:
                r = -dumb_penalty
            else:
                r = -math.log(new_price/last_price)
        if (last_action == 2):
            # rabbit penalty for holding
            r = -1e-5
            if own_btc:
            # hold
                r += math.log(new_price/last_price)
            else:
                r += -math.log(new_price/last_price)
        ## convert to tensor
        return torch.tensor([r], device=self.device)


    def calculate_reward_e(self, new_price, last_price, last_action, own_btc):
        """
        simplified version of reward
        returns -1 for impossible moves and loss, 1 when profit, 0 for other
        """

        if last_action == 0: # buy
            if new_price > last_price:
                # profit from buying
                reward = 1
            else:
                # loss from buying
                reward = -1

        if last_action == 1:
            if new_price > last_price:
                # profit from selling
                reward = 1
            else:
                # loss from selling
                reward = -1


        return torch.tensor([reward], device = self.device)





    # calculate the reward for the last action

    # ???? store the transition in the replay memory

if __name__ == '__main__':
    trainer = Trainer()
    if USE_REPLAY_MEMORY:
        trainer.train_with_rm()
    else:
        trainer.train()

# TODO
# plot with tensorboard
# log score (num gains)
