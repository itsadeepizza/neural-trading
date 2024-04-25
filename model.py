import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device {device}")

learning_rate = 0.05
tanh = torch.nn.Tanh()

class LSTM_CELL(torch.nn.Module):
    """
    see https://colah.github.io/posts/2015-08-Understanding-LSTMs/ for reference
    """

    def __init__(self, state_size, input_size):
        """
        LSTM_CELL constructor
        
        Parameters:    
        context_size: the size of the context vector
        input_size: the size of the input vector
        """
    
        super(LSTM_CELL, self).__init__()
        self.context_size = state_size
        concatenated_size = state_size + input_size
        self.forget_network = torch.nn.Linear(concatenated_size, state_size).to(device)
        self.input_network = torch.nn.Linear(concatenated_size, state_size).to(device)
        self.output_network = torch.nn.Linear(concatenated_size, state_size).to(device)
        self.cell_network = torch.nn.Linear(concatenated_size, state_size).to(device)


    def forward(self, cell_state_prev, hypotesys, x):

        # Step 1
        # The first step in our LSTM is to decide what information we’re going to throw away from
        # the cell state. This decision is made by a sigmoid layer called the “forget gate layer.”
        concatenated_input = torch.cat((hypotesys,x))
        forget_gate = self.forget_network(concatenated_input)
        forget_gate = torch.sigmoid(forget_gate)

        # Step 2
        # The next step is to decide what new information we’re going to store in the cell state.
        input_gate = self.input_network(concatenated_input)
        input_gate = torch.sigmoid(input_gate)
        cell_state_hat = self.cell_network(concatenated_input)
        cell_state_hat = tanh(cell_state_hat)

        # Step 3
        # It’s now time to update the old cell state, Ct−1, into the new cell state Ct.
        new_cell_state = forget_gate * cell_state_prev + input_gate * cell_state_hat

        # Step 4
        # Finally, we need to decide what we’re going to output. This output will be based on our cell state,
        # but will be a filtered version. First, we run a sigmoid layer which decides what parts of the cell state we’re going to output.
        output_gate = self.output_network(concatenated_input)
        output_gate = torch.sigmoid(output_gate)
        new_hypotesis = tanh(new_cell_state) * output_gate

        return (new_hypotesis, new_cell_state)

if __name__ == "__main__":
    lstm = LSTM_CELL(10, 5).to(device)