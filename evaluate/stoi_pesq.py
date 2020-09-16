import torch
from functools import partial
from .metric import compute_pesq
from .metric import compute_stoi

def compute_metric(data):
    clean, pred = data['clean'], data['pred']
    stoi = compute_stoi.compute_stoi(clean, pred)
    pesq = compute_pesq.compute_pesq(clean, pred)
    data.update({'stoi':stoi, 'pesq':pesq})
    return data

class Evaluator():
    def __init__(self, datas, num_workers):
        self.datas = datas
        self.num_workers = num_workers
        self.compute_metric = compute_metric

    # def compute_metric(self, data):
    #     clean, pred = data['clean'], data['pred']
    #     stoi = compute_stoi.compute_stoi(clean, pred)
    #     pesq = compute_pesq.compute_pesq(clean, pred)
    #     data.update({'stoi':stoi, 'pesq':pesq})
    #     return data

    def __iter__(self):
        with torch.multiprocessing.Pool(self.num_workers) as pool:
            for m in pool.map(self.compute_metric, self.datas):
                yield m

    def __len__(self):
        return len(self.datas)

# if __name__ == '__main__':
#     import numpy as np
#     pcms = np.arange(20)
#     print(pcms)
#     eval = Evaluator(pcms, 10)
#     for e in eval:
#         print(e)