import copy

class Metric_Recoder(object):
    def __init__(self, kpi='f1'):
        self.kpi_name = kpi
        self.best_kpi = 0
        self.best_epoch = 0
        self.best_model_state_dict = None

    def update(self, kpi, epoch, model, **kwargs):
        if kpi > self.best_kpi:
            self.best_kpi = kpi
            self.best_epoch = epoch
            if model is not None:
                model_state_dict = copy.deepcopy(model.state_dict())
                self.best_model_state_dict = model_state_dict
            self.best_kwargs = kwargs
    
    def get_best(self):
        return {'epoch': self.best_epoch, self.kpi_name: self.best_kpi, **self.best_kwargs}
    
    def get_best_model_state_dict(self):
        assert self.best_model_state_dict is not None
        return self.best_model_state_dict

if __name__ == '__main__':
    recoder = Metric_Recoder()
    recoder.update(0.9, 1, None, a=1, b=2)
    recoder.update(0.8, 2, None, a=2, b=3)
    recoder.update(0.95, 3, None, a=3, b=4)
    print(recoder.best_kpi, recoder.best_epoch, recoder.best_kwargs)
    # 0.95 3 {'a': 3, 'b': 4}
    print(recoder.get_best())

        

