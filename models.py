import torch
from torch.nn import LazyLinear
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

class ActorCriticGCN:
    class Actor(torch.nn.Module):
        def __init__(self, CktGraph):
            super().__init__()
            self.num_node_features = CktGraph.num_node_features
            self.action_dim = CktGraph.action_dim
            self.device = CktGraph.device
            self.edge_index = CktGraph.edge_index
            self.num_nodes = CktGraph.num_nodes
    
            self.in_channels = self.num_node_features
            self.out_channels = self.action_dim
            self.conv1 = GCNConv(self.in_channels, 32)
            self.conv2 = GCNConv(32, 32)
            self.conv3 = GCNConv(32, 16)
            self.conv4 = GCNConv(16, 16)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state):
            if len(state.shape) == 2:  # if it is not batched graph data (only one data)
                state = state.reshape(1, state.shape[0], state.shape[1])
    
            batch_size = state.shape[0]
            edge_index = self.edge_index
            device = self.device
    
            actions = torch.tensor(()).to(device)
            for i in range(batch_size):
                x = state[i]
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))
                x = F.relu(self.conv4(x, edge_index))
                x = self.lin1(torch.flatten(x))
                x = torch.tanh(x).reshape(1, -1)
                actions = torch.cat((actions, x), axis=0)
    
            return actions
    
    class Critic(torch.nn.Module):
        def __init__(self, CktGraph):
            super().__init__()
            self.num_node_features = CktGraph.num_node_features
            self.action_dim = CktGraph.action_dim
            self.device = CktGraph.device
            self.edge_index = CktGraph.edge_index
            self.num_nodes = CktGraph.num_nodes
    
            self.in_channels = self.num_node_features + self.action_dim
            self.out_channels = 1
            self.conv1 = GCNConv(self.in_channels, 32)
            self.conv2 = GCNConv(32, 32)
            self.conv3 = GCNConv(32, 16)
            self.conv4 = GCNConv(16, 16)
            self.lin1 = LazyLinear(self.out_channels)
    
        def forward(self, state, action):
            batch_size = state.shape[0]
            edge_index = self.edge_index
            device = self.device
    
            action = action.repeat_interleave(self.num_nodes, 0).reshape(
                batch_size, self.num_nodes, -1)
            data = torch.cat((state, action), axis=2)
    
            values = torch.tensor(()).to(device)
            for i in range(batch_size):
                x = data[i]
                x = F.relu(self.conv1(x, edge_index))
                x = F.relu(self.conv2(x, edge_index))
                x = F.relu(self.conv3(x, edge_index))
                x = F.relu(self.conv4(x, edge_index))
                x = self.lin1(torch.flatten(x)).reshape(1, -1)
                values = torch.cat((values, x), axis=0)
    
            return values