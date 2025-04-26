from typing import List, Optional

import torch
import torch.nn as nn


class C1M2Layer(nn.Module):
    """A single layer of the C1M2 (Consistent Memory Module) architecture.

    Applies gating mechanism to combine inputs and previous state
    using softmax over their linear projection.

    Args:
        input_size: Dimension of input features.
        hidden_dim: Dimension of layer outputs.

        bias: Enables/disables bias for inputs layer.
        gate_bias: Enables/disables bias for gates layer.

        activation_fn: Defines activation function at the end of layer.
    """

    def __init__(self, input_size: int, hidden_dim: int,
                 bias: bool = False, gate_bias: bool = False,
                 activation_fn: nn.Module = nn.ReLU()):
        super().__init__()

        self.input_size = input_size
        self.hidden_dim = hidden_dim

        self.inputs_layer = nn.Linear(input_size, hidden_dim, bias=bias)
        self.gates_layer = nn.Linear(hidden_dim * 2, hidden_dim * 2, bias=gate_bias)

        self.activation_fn = activation_fn

    def get_gates(self, inputs: torch.Tensor, state: torch.Tensor):
        """Computes gating coefficients for inputs and previous state.

        Args:
            inputs (torch.Tensor): Linearly projected inputs of shape [batch_size, hidden_dim].
            state (torch.Tensor): Previous layer state of shape [batch_size, hidden_dim].

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - input_gate: Gating weights for new inputs (range [0, 1])
                - state_gate: Gating weights for previous state (range [0, 1])

            Both tensors have shape [batch_size, hidden_dim].
        """
        batch_size, hidden_dim = state.shape

        # Concatinate inputs and state together
        combined = torch.cat([inputs, state], dim=-1)

        # Calculate linear projection of inputs and state
        # and reshape it from [batch_size, hidden_dim * 2] to [batch_size, hidden_dim, 2]
        gate_logits = self.gates_layer(combined).view(batch_size, hidden_dim, 2)

        # Get normalized gate values by applying softmax
        gates = torch.softmax(gate_logits, dim=-1)

        # Split result into two individual gate vectors of size [batch_size, hidden_dim]
        input_gate, state_gate = gates.unbind(dim=-1)

        return input_gate.squeeze(-1), state_gate.squeeze(-1)

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        """Performs forward pass through current layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_size].
            state (torch.Tensor): Previous state tensor of shape [batch_size, hidden_dim].

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - output: Activated output of shape [batch_size, hidden_dim]
                - state: New state tensor of shape [batch_size, hidden_dim]
        """
        batch_size, _ = x.shape

        # Calculate inputs projection
        in_proj = self.inputs_layer(x)

        # Get normalized gates for input projection and state
        in_gate, state_gate = self.get_gates(in_proj, state)

        # Apply gates and add results together, forming new state of layer
        state = state * state_gate + in_proj * in_gate

        # Activate current state and pass it as output
        output = self.activation_fn(state)

        return output, state


class C1M2(nn.Module):
    """Consistent Memory Module (C1M2) implementation.

    A recurrent architecture with softmax-based gating for sequential data processing.

    Args:
        input_size (int): Dimension of input features.
        hidden_dim (int): Hidden state dimension for all layers.

        num_layers (int): Number of sequential C1M2 layers. Default: 1.

        bias (bool): Whether to use bias in linear projections. Default: False.
        gate_bias (bool): Whether to use bias in gating mechanism. Default: False.

        activation_fn (nn.Module): Activation function of each layer. Default: ReLU.
    """

    def __init__(self, input_size: int, hidden_dim: int, num_layers: int = 1,
                 bias: bool = False, gate_bias: bool = False,
                 activation_fn: nn.Module = nn.ReLU()):
        super().__init__()
        self.layers = nn.ModuleList()

        self.hidden_dim = hidden_dim

        for i in range(num_layers):
            layer_inputs = input_size if i == 0 else hidden_dim
            self.layers.append(C1M2Layer(layer_inputs, hidden_dim, bias, gate_bias, activation_fn))

    def step(self, x: torch.Tensor, states: List[torch.Tensor]):
        """Processes a single timestep through all layers.

        Args:
            x (torch.Tensor): Input tensor for current timestep [batch_size, input_size].
            states (List[torch.Tensor]): List of layer states, each [batch_size, hidden_dim].

        Returns:
            torch.Tensor: Output from the last layer [batch_size, hidden_dim].
        """
        for i, layer in enumerate(self.layers):
            x, states[i] = layer(x, states[i])

        return x

    def forward(self, x: torch.Tensor, states: Optional[List[torch.Tensor]] = None):
        """Processes input sequence through all layers.

        Args:
            x (torch.Tensor): Input sequence tensor [batch_size, seq_length, input_size].
            states (List[torch.Tensor], optional): Initial states for each layer.
                If None, zeros are used. Each state has shape [batch_size, hidden_dim].

        Returns:
            tuple[torch.Tensor, List[torch.Tensor]]:
                - outputs: Sequence of outputs [batch_size, seq_length, hidden_dim]
                - states: Final states for each layer (list of [batch_size, hidden_dim] tensors)
        """
        batch_size, seq_len, _ = x.shape

        # If passed states are None then initialize them as zeros for each layer
        if states is None:
            states = [torch.zeros_like(x[:, 0, 0]).unsqueeze(-1).repeat(1, layer.hidden_dim)
                      for layer in self.layers]

        # Create empty tensor for outputs
        outputs = torch.empty(batch_size, seq_len, self.hidden_dim, device=x.device)

        # Iterate over given sequence and pass each step through all CxMy layer (VERY SLOW!!!)
        for i in range(seq_len):
            outputs[:, i] = self.step(x[:, i], states)

        return outputs, states
