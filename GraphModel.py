import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
from torch_geometric.nn import SAGEConv, BatchNorm, global_add_pool
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd

def load_graph_from_gml(file_path):
    graph = nx.read_gml(file_path)
    return graph

def prepare_data(graph):
    data = from_networkx(graph)

    keywords_list = [node_data['keywords'] for _, node_data in graph.nodes(data=True)]
    unique_keywords = sorted(set(kw for kws in keywords_list for kw in kws))
    keyword_to_index = {kw: i for i, kw in enumerate(unique_keywords)}

    keywords_indexed = [[keyword_to_index[kw] for kw in kws] for kws in keywords_list]
    max_keyword_length = max(len(kws) for kws in keywords_indexed)
    keywords_padded = [kws + [0] * (max_keyword_length - len(kws)) for kws in keywords_indexed]
    data.x = torch.tensor(keywords_padded, dtype=torch.float)

    data.y = torch.tensor([node_data['repo_label'] for _, node_data in graph.nodes(data=True)], dtype=torch.long)
    
    return data

def split_data(data):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_idx, test_idx = next(skf.split(np.zeros(len(data.y)), data.y))
    val_idx, test_idx = train_test_split(test_idx, test_size=0.5, random_state=42, stratify=data.y[test_idx])

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
    base_name = file_path.split('.')[0]

    graph = load_graph_from_gml(file_path)
    data = prepare_data(graph)
    data = split_data(data)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GraphSAGE(data.num_features, 64, 128, len(data.y.unique())).to(device)

    data = data.to(device)
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    data.train_mask = data.train_mask.to(device)
    data.val_mask = data.val_mask.to(device)
    data.test_mask = data.test_mask.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, verbose=True)

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
        logits = model(data.x, data.edge_index)
        accs_exact = []
        accs_top3 = []
        preds_exact = []
        preds_top3 = []
        targets = []
        for mask in [data.train_mask, data.val_mask, data.test_mask]:
            # Exact prediction
            pred_exact = logits[mask].max(1)[1]
            acc_exact = pred_exact.eq(data.y[mask]).sum().item() / mask.sum().item()
            accs_exact.append(acc_exact)

            top3_pred = logits[mask].topk(3, dim=1).indices
            correct_top3 = top3_pred.eq(data.y[mask].unsqueeze(1)).sum(1)
            acc_top3 = correct_top3.sum().item() / mask.sum().item()
            accs_top3.append(acc_top3)

            preds_exact.append(pred_exact.cpu().numpy())
            preds_top3.append(top3_pred.cpu().numpy())
            targets.append(data.y[mask].cpu().numpy())

        return accs_exact[0], accs_exact[1], accs_exact[2], accs_top3[1], accs_top3[2], preds_exact[1], preds_top3[1], targets[1], preds_exact[2], preds_top3[2], targets[2]

    train_losses = []
    val_accuracies_exact = []
    test_accuracies_exact = []
    val_accuracies_top3 = []
    test_accuracies_top3 = []
    best_val_acc = 0
    patience = 50
    patience_counter = 0

    for epoch in range(1000):
        loss = train()
        train_acc_exact, val_acc_exact, test_acc_exact, val_acc_top3, test_acc_top3, val_preds_exact, val_preds_top3, val_targets, test_preds_exact, test_preds_top3, test_targets = test()
        train_losses.append(loss)
        val_accuracies_exact.append(val_acc_exact)
        test_accuracies_exact.append(test_acc_exact)
        val_accuracies_top3.append(val_acc_top3)
        test_accuracies_top3.append(test_acc_top3)
        
        print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Train Acc Exact: {train_acc_exact:.4f}, Val Acc Exact: {val_acc_exact:.4f}, Test Acc Exact: {test_acc_exact:.4f}')
        print(f'              Val Acc Top3: {val_acc_top3:.4f}, Test Acc Top3: {test_acc_top3:.4f}')

        if val_acc_exact > best_val_acc:
            best_val_acc = val_acc_exact
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        scheduler.step(loss)

    # Save Loss and Accuracy Data
    metrics_data = {
        'Epoch': list(range(1, len(train_losses) + 1)),
        'Train Loss': train_losses,
        'Validation Accuracy Exact': val_accuracies_exact,
        'Test Accuracy Exact': test_accuracies_exact,
        'Validation Accuracy Top3': val_accuracies_top3,
        'Test Accuracy Top3': test_accuracies_top3
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_df.to_csv(f'{base_name}_metrics.csv', index=False)

    # Plot and Save Loss and Accuracy Graphs
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(16, 4))
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, 'r', label='Training loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'{base_name}_training_loss.png')

    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_accuracies_exact, 'b', label='Validation Accuracy Exact')
    plt.plot(epochs, test_accuracies_exact, 'g', label='Test Accuracy Exact')
    plt.title('Validation and Test Accuracy Exact')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{base_name}_accuracy_exact.png')

    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_accuracies_top3, 'b', label='Validation Accuracy Top3')
    plt.plot(epochs, test_accuracies_top3, 'g', label='Test Accuracy Top3')
    plt.title('Validation and Test Accuracy Top3')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(f'{base_name}_accuracy_top3.png')
    plt.show()

    cm_val_exact = confusion_matrix(val_targets, val_preds_exact)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm_val_exact, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix - Validation Set (Exact)')
    plt.axis('off')
    plt.savefig(f'{base_name}_confusion_matrix_val_exact.png')
    plt.show()

    cm_val_top3 = confusion_matrix(val_targets, [p[0] for p in val_preds_top3])  # Use first prediction for confusion matrix
    plt.figure(figsize=(8, 8))
    plt.imshow(cm_val_top3, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix - Validation Set (Top3)')
    plt.axis('off')
    plt.savefig(f'{base_name}_confusion_matrix_val_top3.png')
    plt.show()

    cm_test_exact = confusion_matrix(test_targets, test_preds_exact)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm_test_exact, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix - Test Set (Exact)')
    plt.axis('off')
    plt.savefig(f'{base_name}_confusion_matrix_test_exact.png')
    plt.show()

    cm_test_top3 = confusion_matrix(test_targets, [p[0] for p in test_preds_top3])  # Use first prediction for confusion matrix
    plt.figure(figsize=(8, 8))
    plt.imshow(cm_test_top3, cmap='Blues')
    plt.colorbar()
    plt.title('Confusion Matrix - Test Set (Top3)')
    plt.axis('off')
    plt.savefig(f'{base_name}_confusion_matrix_test_top3.png')
    plt.show()

    f1_exact = f1_score(test_targets, test_preds_exact, average='weighted')
    precision_exact = precision_score(test_targets, test_preds_exact, average='weighted')
    recall_exact = recall_score(test_targets, test_preds_exact, average='weighted')

    f1_top3 = f1_score(test_targets, [p[0] for p in test_preds_top3], average='weighted')
    precision_top3 = precision_score(test_targets, [p[0] for p in test_preds_top3], average='weighted')
    recall_top3 = recall_score(test_targets, [p[0] for p in test_preds_top3], average='weighted')

    additional_metrics = {
        'Metric': ['F1 Score', 'Precision', 'Recall'],
        'Exact': [f1_exact, precision_exact, recall_exact],
        'Top3': [f1_top3, precision_top3, recall_top3]
    }
    additional_metrics_df = pd.DataFrame(additional_metrics)
    additional_metrics_df.to_csv(f'{base_name}_additional_metrics.csv', index=False)

    print(f'Exact - F1 Score: {f1_exact:.4f}, Precision: {precision_exact:.4f}, Recall: {recall_exact:.4f}')
    print(f'Top3 - F1 Score: {f1_top3:.4f}, Precision: {precision_top3:.4f}, Recall: {recall_top3:.4f}')
