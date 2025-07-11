import numpy as np
from torch import Tensor
from captum._utils.models.model import Model
import torch.nn as nn

class LemnaModel(Model):
    def __init__(self):
        pass
    
    def fit(self, train_data):
        # Reference: https://github.com/Henrygwb/Explaining-DL/blob/master/lemna/code/lemna_beta.py#L113
        from rpy2 import robjects
        from rpy2.robjects.packages import importr
        import rpy2.robjects.numpy2ri
        r = robjects.r
        rpy2.robjects.numpy2ri.activate()
        # should install these two R packages before running
        importr('genlasso')
        importr('gsubfn')

        num_batches = 0
        xs, ys, ws = [], [], []
        for data in train_data:
            if len(data) == 3:
                x, y, w = data
            else:
                assert len(data) == 2
                x, y = data
                w = None

            xs.append(x.cpu().numpy())
            ys.append(y.cpu().numpy())
            if w is not None:
                ws.append(w.cpu().numpy())
            num_batches += 1
        
        x = np.concatenate(xs, axis=0)
        y = np.concatenate(ys, axis=0)
        if len(ws) > 0:
            w = np.concatenate(ws, axis=0)
        else:
            w = None

        # training using x, y, w
        # print(x.shape, y.shape, w.shape if w is not None else None)  # for debug if needed
        y = y.reshape(-1, 1)  
        w = w.reshape(-1, 1) 

        data_explain = x
        label_sampled = y
        X = r.matrix(data_explain, nrow=data_explain.shape[0], ncol=data_explain.shape[1])
        Y = r.matrix(label_sampled, nrow=label_sampled.shape[0], ncol=label_sampled.shape[1])

        n = r.nrow(X)
        p = r.ncol(X)
        results = r.fusedlasso1d(y=Y, X=X)
        self.fusedlasso_result = np.array(r.coef(results, np.sqrt(n * np.log(p)))[0])[:, -1]  # shape: (feature_num,)

        return {"train_time": 0.5}
    
    def __call__ (self,x:Tensor)->Tensor:
        raise NotImplementedError("This method should not be called. ")
        
    def representation(self) -> Tensor:
        r"""
        Returns a tensor which describes the hyper-plane input space. This does
        not include the bias. For bias/intercept, please use `self.bias`
        """
        assert hasattr(self, "fusedlasso_result"), "Model has not been trained yet"
        return Tensor(self.fusedlasso_result)
    