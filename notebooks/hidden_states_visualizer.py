

import numpy as np
import pandas as pd
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

dataset = "kp20k"

# 1. seq2set
model_type = "bertbase_seq2set_kp20k"
save_file = "tsne_plot1.png"

# 2. proposed model
# model_type = "bertbase_seq2set(proposed_phi_0.1)_kp20k"
# save_file = "tsne_plot2.png"

data_path = f"checkpoints/phase5_exp/abstractive_models/{model_type}/metrics_4(Greedy_HSrun)_{dataset}_predictions_4_hiddenstates.pickle"
num_samples = 5
offset = num_samples*6

all_records = []
with open(data_path, 'rb') as f:
    hs_records = pickle.load(f)
    max_kps = 5
    max_seq_len = 7
    
    for record in tqdm(hs_records, total=len(hs_records)):
        bs, seq_len, hdim = record.size()
        record_m = record.reshape((bs, max_kps, max_seq_len, hdim))[:,:,:2,:].mean(dim=2)
        all_records.append(record_m.detach().to('cpu'))

truncate = 10
record_m = torch.cat(all_records[:truncate], dim=0)
print(record_m.shape)
bs = record_m.size()[0]

# Reshape the tensor to [320, 768] by merging the batch and embedding type dimensions
reshaped_tensor = record_m.reshape(-1, 768)

# Define labels for each data point based on the embedding type (0-4)
# labels = np.repeat(np.arange(5), bs)

# Perform T-SNE dimensionality reduction
tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42)
tsne_result = tsne.fit_transform(reshaped_tensor)

# Create a scatter plot to visualize the T-SNE results with different colors for each embedding type
plt.figure(figsize=(8, 6))
# scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=labels, cmap='viridis', alpha=0.7)

scatter = plt.scatter(tsne_result[offset:num_samples+offset, 0], tsne_result[offset:num_samples+offset, 1], alpha=0.5)
plt.title("T-SNE Visualization of Different Embedding Types")
# plt.colorbar(scatter)
# plt.show()
plt.savefig(save_file)
