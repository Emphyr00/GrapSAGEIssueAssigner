import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.nn import SAGEConv, BatchNorm, global_add_pool
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split

def load_graph_from_gml(file_path):
    graph = nx.read_gml(file_path)
    return graph

def prepare_data(graph):
    data = from_networkx(graph)

    # Ensure all keywords are properly processed as indices
    keywords_list = [node_data['keywords'] for _, node_data in graph.nodes(data=True)]
    unique_keywords = sorted(set(kw for kws in keywords_list for kw in kws))
    keyword_to_index = {kw: i for i, kw in enumerate(unique_keywords)}

    keywords_indexed = [[keyword_to_index[kw] for kw in kws] for kws in keywords_list]
    max_keyword_length = max(len(kws) for kws in keywords_indexed)
    keywords_padded = [kws + [0] * (max_keyword_length - len(kws)) for kws in keywords_indexed]
    data.x = torch.tensor(keywords_padded, dtype=torch.float)

    # Set the label for each node based on the repo label
    data.y = torch.tensor([node_data['repo_label'] for _, node_data in graph.nodes(data=True)], dtype=torch.long)
    
    return data

def split_data(data):
    train_idx, test_idx = train_test_split(np.arange(data.num_nodes), test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42)

    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask

    return data

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels1, hidden_channels2, out_channels):
        super(GraphSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels1)
        self.bn1 = BatchNorm(hidden_channels1)
        self.conv2 = SAGEConv(hidden_channels1, hidden_channels2)
        self.bn2 = BatchNorm(hidden_channels2)
        self.conv3 = SAGEConv(hidden_channels2, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)

def train_and_test(file_path):
    graph = load_graph_from_gml(file_path)
    data = prepare_data(graph)
    data = split_data(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(data.num_features, 256, 64, len(data.y.unique())).to(device)

    # Move data to device
    data = data.to(device)
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)
    data.test_mask = data.test_mask.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, verbose=True)

    def train():
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    def test():
        model.eval()
        logits, accs = model(data.x, data.edge_index), []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs.append(acc)
        return accs

    best_val_acc = 0
    patience = 50
    for epoch in range(1000):
        loss = train()
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        scheduler.step(loss)

# Example usage:
# train_and_test('keyword_graph_final4.gml')
