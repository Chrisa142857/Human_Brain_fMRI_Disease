import torch

def bold_signal_to_trends(bold: torch.Tensor):
    '''
        bold.shape: [timeseries, roi_num]
    '''
    next = bold[1:]
    pre = bold[:-1]
    out = torch.zeros(bold.shape[0]-1, bold.shape[1])
    out[next > pre] = 1
    out[next == pre] = 0
    out[next < pre] = -1
    return out

def bold_signal_threshold(bold: torch.Tensor, thr=0.5, topk=5):
    '''
        bold.shape: [timeseries, roi_num]
    '''
    out = torch.zeros(bold.shape[0], bold.shape[1])
    thr1 = bold.T.topk(topk)[0].mean(1) * thr # roi_num
    thr2 = (-1*bold.T).topk(topk)[0].mean(1) * thr * -1# roi_num
    for i in range(bold.shape[1]):
        out[bold[:, i] > thr1[i], i] = 1
        if thr2[i] < 0:
            out[bold[:, i] < thr2[i], i] = -1
    return out
