import torch
import torch.nn as nn
from packages.utils.utils import calculate_laplacian_with_self_loop
import argparse

class TGCNGraphConvolutionAttention(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, bias: float = 0.0):
        super(TGCNGraphConvolutionAttention, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + 1, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + 1) * batch_size)
        )
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + 1, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + 1)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGCNCellAttention(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TGCNCellAttention, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolutionAttention(
            self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = TGCNGraphConvolutionAttention(
            self.adj, self._hidden_dim, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}

class SelfAttention(nn.Module):
    def __init__(self, num_nodes, output_dim: int, bias: float = 0.0):
      super(SelfAttention, self).__init__()
      self.num_nodes = num_nodes
      self._num_gru_units = output_dim
      self._output_dim = output_dim
      self._bias_init_value = bias

      # SelfAttention Weights Initialization
      self.weights_att1 = nn.Parameter(
          torch.FloatTensor(self._num_gru_units, 1))
      self.weights_att2 = nn.Parameter(
          torch.FloatTensor(self.num_nodes, 1))
      self.weights_att3 = nn.Parameter(
          torch.FloatTensor(self.num_nodes, 1))
      self.weights_att4 = nn.Parameter(
          torch.FloatTensor(self.num_nodes, 1))
      # SelfAttention Biases Initialization
      self.bias_att1 = nn.Parameter(torch.FloatTensor([1]))
      self.bias_att2 = nn.Parameter(torch.FloatTensor([1]))
      self.bias_att3 = nn.Parameter(torch.FloatTensor([1]))
      self.bias_att4 = nn.Parameter(torch.FloatTensor([1]))
      self.reset_parameters()
      # Gamma for Self Attention
      self.gamma = nn.Parameter(torch.FloatTensor([1]))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights_att1)
        nn.init.xavier_uniform_(self.weights_att2)
        nn.init.xavier_uniform_(self.weights_att3)
        nn.init.xavier_uniform_(self.weights_att4)
        nn.init.constant_(self.bias_att1, self._bias_init_value)
        nn.init.constant_(self.bias_att2, self._bias_init_value)
        nn.init.constant_(self.bias_att3, self._bias_init_value)
        nn.init.constant_(self.bias_att4, self._bias_init_value)

    def reset_gamma(self):
      nn.init.constant_(self.gamma, self._bias_init_value)

    def forward(self, hidden_states, gru_units: int, num_nodes: int, seq_len: int):
      # x (batch_size, seq_len, num_nodes, self._hidden_dim)
      hidden_states = torch.reshape(hidden_states, [-1, gru_units]) @ self.weights_att1 + self.bias_att1

      f = torch.reshape(hidden_states, [-1, num_nodes]) @ self.weights_att2 + self.bias_att2
      g = torch.reshape(hidden_states, [-1, num_nodes]) @ self.weights_att3 + self.bias_att3
      h = torch.reshape(hidden_states, [-1, num_nodes]) @ self.weights_att4 + self.bias_att4

      f = torch.reshape(f, [-1,seq_len])
      g = torch.reshape(g, [-1,seq_len])
      h = torch.reshape(h, [-1,seq_len])

      # Attention Matrix
      s = g * f
      beta = nn.functional.softmax(s, dim=-1)

      #o = beta * h
      #self.reset_gamma()

      #hidden_states = o.unsqueeze(2) * self.gamma + hidden_states
      context = beta.unsqueeze(2) * torch.reshape(hidden_states, [-1,seq_len,num_nodes])
      return context, beta


class TGCNAttention(nn.Module):
    def __init__(self, adj, seq_len: int, pre_len: int, hidden_dim: int, **kwargs):
        super(TGCNAttention, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.pre_len = pre_len
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCellAttention(self.adj, self._input_dim, self._hidden_dim)
        self.attention = SelfAttention(self._input_dim, self._hidden_dim)

        self.weights_out = nn.Parameter(torch.FloatTensor(seq_len, pre_len))
        self.biases_out = nn.Parameter(torch.FloatTensor(pre_len))
        self.reset_parameters_out()

    def reset_parameters_out(self):
        nn.init.xavier_uniform_(self.weights_out)
        nn.init.constant_(self.biases_out, 0.0)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes = inputs.shape
        assert self._input_dim == num_nodes

        # GCN and GRU
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(
            inputs)
        hidden_states = torch.zeros(batch_size, seq_len, num_nodes, self._hidden_dim).type_as(
            inputs)
        outputs = torch.zeros(batch_size, seq_len, num_nodes).type_as(inputs)

        # RNN Implementation
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :], hidden_state)
            outputs[:, i, :] = output.reshape((batch_size, num_nodes, self._hidden_dim))[:,:,-1]
            # (batch_size, seq_len, num_nodes)
            hidden_states[:, i, :, :] = hidden_state.reshape((batch_size, num_nodes, self._hidden_dim))

        # Self Attention Implementation
        # (batch_size, seq_len, num_nodes)
        context, alpha = self.attention(hidden_states=hidden_states, gru_units=self._hidden_dim, num_nodes=num_nodes, seq_len=seq_len)
        # (batch_size, num_nodes, seq_len)
        context = context.transpose(2, 1)
        # (batch_size * num_nodes, seq_len)
        atten_output = torch.reshape(context, [-1,seq_len]) @ self.weights_out + self.biases_out
        # (batch_size, num_nodes, pre_len)
        atten_output = torch.reshape(atten_output, [batch_size, num_nodes, self.pre_len])
        # (batch_size, pre_len, num_nodes)
        atten_output = atten_output.transpose(1, 2)
        return atten_output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}