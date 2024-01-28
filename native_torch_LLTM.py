import torch
import math    
import time
from torch.nn import functional as F

class native_torch_LLTM(torch.nn.Module):
    def __init__(self, input_features, state_size):
        super(native_torch_LLTM, self).__init__()
        self.input_features = input_features
        self.state_size = state_size
        # 3 * state_size for input gate, output gate and candidate cell gate.
        # input_features + state_size because we will multiply with [input, h].
        self.weights = torch.nn.Parameter(
            torch.empty(3 * state_size, input_features + state_size))
        self.bias = torch.nn.Parameter(torch.empty(3 * state_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.state_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input, state):
        old_h, old_cell = state
        X = torch.cat([old_h, input], dim=1)

        # Compute the input, output and candidate cell gates with one MM.
        gate_weights = F.linear(X, self.weights, self.bias)
        # Split the combined gate weight matrix into its components.
        gates = gate_weights.chunk(3, dim=1)

        input_gate = torch.sigmoid(gates[0])
        output_gate = torch.sigmoid(gates[1])
        # Here we use an ELU instead of the usual tanh.
        candidate_cell = F.elu(gates[2])

        # Compute the new cell state.
        new_cell = old_cell + candidate_cell * input_gate
        # Compute the new hidden state and output.
        new_h = torch.tanh(new_cell) * output_gate

        return new_h, new_cell

if __name__ == '__main__':
    cuda_device = "cpu" # device object representing GPU
    batch_size = 1024
    input_features = 32
    state_size = 128

    # Note the device=cuda_device arguments here
    X = torch.randn(batch_size, input_features, device=cuda_device)
    h = torch.randn(batch_size, state_size, device=cuda_device)
    C = torch.randn(batch_size, state_size, device=cuda_device)

    rnn = native_torch_LLTM(input_features, state_size).to(cuda_device)

    forward = 0
    backward = 0
    for _ in range(1000):
        start = time.time()
        new_h, new_C = rnn(X, (h, C))
        forward += time.time() - start

        start = time.time()
        (new_h.sum() + new_C.sum()).backward()
        backward += time.time() - start

    print('Forward: {:.3f} us | Backward {:.3f} us'.format(forward * 1e6/1e3, backward * 1e6/1e3))