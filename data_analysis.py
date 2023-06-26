from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets import RoIBOLDCorrCoefWin, RoIBOLD

datatag = 'oasis_bold'
# dataset = RoIBOLDCorrCoefWin(
#     # data_csvn='OASIS3_convert_vs_nonconvert.csv', 
#     data_csvn='ADNI_AAL90_5class.csv', roi_start=0, roi_end=90,
# )
dataset = RoIBOLD(
    data_csvn='OASIS3_convert_vs_nonconvert.csv', 
    # data_csvn='ADNI_AAL90_5class.csv',
)

class_dict = {v: k for k,v in dataset.class_dict.items()}
all_data = []
all_label = []
for di, data in enumerate(dataset):
    label = data[1]
    data = data[0]
    for d in data:
        # triu_ind = torch.triu_indices(d.shape[0], d.shape[1])
        # all_data.append(d[triu_ind[0], triu_ind[1]].numpy())
        all_data.append(d)
        all_label.append(label.numpy())
        
all_data = np.stack(all_data)
all_label = np.stack(all_label)
print(all_data.shape)
X = TSNE(n_components=2, learning_rate='auto', 
                  init='random', perplexity=3).fit_transform(all_data)
print(X.shape)
for label in np.unique(all_label):
    x = X[all_label==label]
    plt.scatter(x[:, 0], x[:, 1], label=class_dict[label])

plt.legend()
plt.axis('off')
plt.tight_layout()
plt.savefig('tsne_%s.jpg' % datatag, dpi=600)
plt.close()