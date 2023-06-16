from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np
import torch, csv, os
from tqdm import trange
STEP_SIZE = 50
WIN_SIZE = 80

class RoIBOLD(Dataset):
    def __init__(self, data_csvn=None, roi_num=191, preproc=None) -> None:
        # step_size = STEP_SIZE
        # seq_len = WIN_SIZE
        with open(data_csvn, 'r') as f:
            lines = f.read().split('\n')[1:-1]
        self.label_dict = {
            l.split(',')[0]: l.split(',')[1] 
        for l in lines}
        self.flist = [
            l.split(',')[3].split('@')
        for l in lines]
        self.labels = []
        # self.seq_len = seq_len
        self.class_dict = {k: classi for classi, k in enumerate(np.unique(list(self.label_dict.values())))}
        self.subject_names = []
        self.subject_id_list = []
        self.data = []
        sid = 0
        for fpaths in tqdm(self.flist, desc="init dataset"):
            data = []
            for fpath in fpaths:
                data.append(np.loadtxt(fpath)[:, :roi_num]) # Time x RoI
            subject_n = fpath.split('/')[-1].split('_')[0]
            data = torch.from_numpy(np.concatenate(data).astype(np.float32))
            if preproc:
                data = preproc(data)
            self.data.append(data)
            self.labels.append(self.class_dict[self.label_dict[subject_n]])
            self.subject_id_list.append(sid)
            if subject_n not in self.subject_names: 
                self.subject_names.append(subject_n)
                sid += 1

        # self.data = torch.from_numpy(np.stack(self.data).astype(np.float32))
        self.labels = torch.LongTensor(self.labels)
        self.subject_id_list = torch.LongTensor(self.subject_id_list)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.subject_id_list[idx]

    def __len__(self):
        return len(self.data)
    
    def collate_fn(self, batch):
        data = []
        labels = []
        subs = []
        for d, label, sub in batch:
            data.append(d)
            labels.append(torch.LongTensor([label]))
            subs.append(torch.LongTensor([sub]))
            # d = torch.from_numpy(d)
            # if d.shape[0] >= self.seq_len:
            #     st = torch.randint(low=0, high=d.shape[0]-self.seq_len, size=(1,))
            #     data.append(d[st:st+self.seq_len])
            # else:
            #     data.append(torch.cat([d, torch.zeros(self.seq_len-d.shape[0], d.shape[1], dtype=d.dtype)], dim=0))
        # data = torch.stack(data)
        return data, torch.cat(labels), torch.cat(subs)


class RoIBOLDCorrCoef(Dataset):
    def __init__(self, data_csvn=None, roi_start=41, roi_end=191, preproc=None) -> None:
        step_size = STEP_SIZE
        seq_len = WIN_SIZE
        self.roi_num = roi_end - roi_start
        with open(data_csvn, 'r') as f:
            lines = f.read().split('\n')[1:-1]
        self.label_dict = {
            l.split(',')[0]: l.split(',')[1] 
        for l in lines}
        self.flist = [
            l.split(',')[3].split('@')
        for l in lines]
        self.labels = []
        self.class_dict = {k: classi for classi, k in enumerate(np.unique(list(self.label_dict.values())))}
        self.subject_names = []
        self.subject_id_list = []
        self.data = []
        sid = 0
        for fpaths in tqdm(self.flist, desc="init dataset"):
            data = []
            for fpath in fpaths:
                data.append(np.loadtxt(fpath)[:, roi_start:roi_end]) # Time x RoI
            subject_n = fpath.split('/')[-1].split('_')[0]
            data = torch.from_numpy(np.concatenate(data).astype(np.float32))
            if preproc:
                data = preproc(data)
            data = torch.stack([corrcoef(data[st:st+seq_len]) for st in range(0, len(data), step_size)])
            data = data[~data.isnan().any(1).any(1)]
            torch.nn.init.orthogonal_(data)
            data = data + 1e-10*torch.stack([torch.eye(data.shape[-1]) for _ in range(len(data))], -1).permute(2,0,1)
            for d in data:
                self.data.append(d)
                self.labels.append(self.class_dict[self.label_dict[subject_n]])
                self.subject_id_list.append(sid)
            if subject_n not in self.subject_names: 
                self.subject_names.append(subject_n)
                sid += 1

        self.labels = torch.LongTensor(self.labels)
        self.subject_id_list = torch.LongTensor(self.subject_id_list)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.subject_id_list[idx]

    def __len__(self):
        return len(self.data)
        
    def collate_fn(self, batch):
        data = []
        labels = []
        subs = []
        for d, label, sub in batch:
            data.append(d)
            labels.append(torch.LongTensor([label]))
            subs.append(torch.LongTensor([sub]))
        return torch.stack(data), torch.cat(labels), torch.cat(subs)


class RoIBOLDCorrCoefWin(Dataset):
    def __init__(self, data_csvn=None, roi_start=41, roi_end=191, preproc=None) -> None:
        step_size = STEP_SIZE
        seq_len = WIN_SIZE
        self.roi_num = roi_end - roi_start
        with open(data_csvn, 'r') as f:
            lines = f.read().split('\n')[1:-1]
        self.label_dict = {
            l.split(',')[0]: l.split(',')[1] 
        for l in lines}
        self.flist = [
            l.split(',')[3].split('@')
        for l in lines]
        self.labels = []
        self.seq_len = seq_len
        self.class_dict = {k: classi for classi, k in enumerate(np.unique(list(self.label_dict.values())))}
        self.subject_names = []
        self.subject_id_list = []
        self.data = []
        sid = 0
        for fpaths in tqdm(self.flist, desc="init dataset"):
            data = []
            for fpath in fpaths:
                data.append(np.loadtxt(fpath)[:, roi_start:roi_end]) # Time x RoI
            subject_n = fpath.split('/')[-1].split('_')[0]
            data = torch.from_numpy(np.concatenate(data).astype(np.float32))
            if preproc:
                data = preproc(data)
            self.data.append([data[st:st+seq_len] for st in range(0, len(data), step_size)])
            self.labels.append(self.class_dict[self.label_dict[subject_n]])
            self.subject_id_list.append(sid)
            if subject_n not in self.subject_names: 
                self.subject_names.append(subject_n)
                sid += 1

        for i in trange(len(self.data), desc='Get corr coef SPD matrix'):
            self.corrcoef_spd(i)
        self.labels = torch.LongTensor(self.labels)
        self.subject_id_list = torch.LongTensor(self.subject_id_list)

    def corrcoef_spd(self, idx):
        data = torch.stack([corrcoef(d.T) for d in self.data[idx]])
        data = data[~data.isnan().any(1).any(1)]
        # torch.nn.init.orthogonal_(data)
        # data = data + 1e-10*torch.stack([torch.eye(data.shape[-1]) for _ in range(len(data))], -1).permute(2,0,1)
        self.data[idx] = data

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx], self.subject_id_list[idx]

    def __len__(self):
        return len(self.data)
        
    def collate_fn(self, batch):
        data = []
        labels = []
        subs = []
        for d, label, sub in batch:
            data.append(d)
            labels.append(torch.LongTensor([label]))
            subs.append(torch.LongTensor([sub]))
        return data, torch.cat(labels), torch.cat(subs)


def corrcoef(X):
    avg = torch.mean(X, dim=-1)
    X = X - avg[..., None]
    X_T = X.swapaxes(-2, -1)
    c = torch.matmul(X, X_T)
    d = torch.diagonal(c, 0, -2, -1)
    stddev = torch.sqrt(torch.tensor(d))
    c = c / stddev[..., None]
    c = c / stddev[..., None, :]
    c = torch.clip(c, -1, 1, out=c)
    return c
    # # ChatGPT answers:
    # # 计算相关性系数矩阵
    # # 输入:
    # #   X: 输入数据，形状为 (num_samples, num_variables)
    # # 输出:
    # #   corr_matrix: 相关性系数矩阵，形状为 (num_variables, num_variables)
    # # 计算均值
    # mean_X = torch.mean(X, dim=0)
    # # 计算标准差
    # std_X = torch.std(X, dim=0)
    # # 归一化数据
    # X_normalized = (X - mean_X) / std_X
    # # 计算协方差矩阵
    # cov_matrix = torch.matmul(X_normalized.T, X_normalized) / X.shape[0]
    # # 计算相关性系数矩阵
    # corr_matrix = cov_matrix / torch.sqrt(torch.diag(cov_matrix))
    # return corr_matrix


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dataset = RoIBOLDCorrCoefWin(
        data_csvn='OASIS3_convert_vs_nonconvert.csv', 
    )
    for di, data in enumerate(dataset):
        fig, ax = plt.subplots(5,8)
        ax = ax.reshape(-1)
        for i in range(40):
            if i >= len(data[0]): break
            ax[i].matshow(data[0][i])
        plt.savefig('CCmats/%d_%d.jpg' % (di,data[1]), dpi=600)
        plt.close()
        