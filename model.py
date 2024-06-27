import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device {device}")

learning_rate = 0.05


class LSTM_CELL(torch.nn.Module):
    """
    see https://colah.github.io/posts/2015-08-Understanding-LSTMs/ for reference
    """

    def __init__(self, state_size, input_size):
        """
        LSTM_CELL constructor
        
        Parameters:    
        state_size: the size of the cell state vector
        input_size: the size of the input vector
        """

        super(LSTM_CELL, self).__init__()
        # self.context_size = state_size
        concatenated_size = state_size + input_size


        self.forget_network = torch.nn.Sequential(
                        torch.nn.Linear(concatenated_size, 100),
                        torch.nn.ReLU(),
                        torch.nn.Linear(100, 50),
                        torch.nn.ReLU(),
                        torch.nn.Linear(50, state_size),
                        torch.nn.Sigmoid()
        )

        self.input_network = torch.nn.Sequential(
                        torch.nn.Linear(concatenated_size, 100),
                        torch.nn.ReLU(),
                        torch.nn.Linear(100, 50),
                        torch.nn.ReLU(),
                        torch.nn.Linear(50, state_size),
                        torch.nn.Sigmoid()
        )

        self.cell_network = torch.nn.Sequential(
                        torch.nn.Linear(concatenated_size, 100),
                        torch.nn.ReLU(),
                        torch.nn.Linear(100, 50),
                        torch.nn.ReLU(),
                        torch.nn.Linear(50, state_size),
                        torch.nn.Tanh()
        )

        self.output_network = torch.nn.Sequential(
                        torch.nn.Linear(concatenated_size, 100),
                        torch.nn.ReLU(),
                        torch.nn.Linear(100, 50),
                        torch.nn.ReLU(),
                        torch.nn.Linear(50, state_size),
                        torch.nn.Sigmoid()
        )

    def forward(self, cell_state_prev, hypotesis, x):
        concatenated_input = torch.cat((hypotesis, x))

        # Step 1
        # The first step in our LSTM is to decide what information we’re going to throw away from
        # the cell state. This decision is made by a sigmoid layer called the “forget gate layer.”
        forget_gate = self.forget_network(concatenated_input)

        # Step 2
        # The next step is to decide what new information we’re going to store in the cell state.
        input_gate = self.input_network(concatenated_input)
        cell_state_hat = self.cell_network(concatenated_input)

        # Step 3
        # It’s now time to update the old cell state, Ct−1, into the new cell state Ct.
        new_cell_state = (forget_gate * cell_state_prev) + (input_gate * cell_state_hat)

        # Step 4
        # Finally, we need to decide what we’re going to output. This output will be based on our cell state,
        # but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output.
        output_gate = self.output_network(concatenated_input)
        new_hypotesis = torch.tanh(new_cell_state) * output_gate

        return (new_hypotesis, new_cell_state)


class LSTM_Trader(torch.nn.Module):
    """
    see https://colah.github.io/posts/2015-08-Understanding-LSTMs/ for reference
    """

    def __init__(self, state_size, input_size, output_size):
        super(self).__init__()
        self.lstm = LSTM_CELL(state_size, input_size)
        self.fc = torch.nn.Linear(state_size, output_size)

    def forward(self, cell_state_prev, hypotesys, x):
        hypotesis, cell_state = self.lstm(cell_state_prev, hypotesys, x)
        output = self.fc(hypotesis)
        output = torch.sigmoid(output)

        return (output, hypotesis, cell_state)


if __name__ == "__main__":
    state_size = 20
    input_size = 2  # (size of X_t for any choosen t)
    output_size = 3  # hypotesis( dim = state_size) -> fc -> output( dim = output_size)

    lstm = LSTM_CELL(state_size, input_size)  # NOTE is todevice needed?

    # lil test loop

    N_SAMPLES = 10
    random_input = torch.randn((10, 2))

    state = torch.randn(state_size)
    hypotesis = torch.randn(state_size)

    with torch.no_grad():
        for n in range(N_SAMPLES):
            (hypotesis, state) = lstm(state, hypotesis, random_input[n])
            print(hypotesis)
