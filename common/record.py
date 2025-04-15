import torch
import os
from copy import deepcopy

class Record():

    def __init__(self, 
                 path: str,
                 train_dataset, valid_dataset, test_dataset, 
                 model, trainer,
                 record_name='record'
                ) -> None:
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.test_dataset = test_dataset
        self.model = model
        self.trainer = trainer
        self.path = path
        self.record_name = record_name
        self.checkpoints = {}
    
    def save(self, record_name=None) -> None:
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        
        record_name = record_name if record_name else self.record_name
        
        torch.save(self, os.path.join(self.path, f"{record_name}.pt"))
    
    @staticmethod
    def load(path: str, map_location='cpu') -> 'Record':
        record = torch.load(path, map_location=map_location, weights_only=False) #Keep old behaviour, fix with `torch.serialization.add_safe_globals(
        #Backward compatibility
        if not hasattr(record, 'checkpoints'):
            record.checkpoints = {}
        return record

class Checkpoint():

    def __init__(self, recorder, save_fn, checkpoint_interval) -> None:
        self.recorder = recorder
        self.best_acc = 0
        self.save = save_fn
        self.ci = checkpoint_interval
    
    def checkpoint(self, epoch, test_metrics):
        s = False
        if test_metrics['acc'] > self.best_acc:
            self.best_acc = test_metrics['acc']
            self.recorder.checkpoints['best'] = deepcopy(self.recorder.model.state_dict())
            s = True
        if self.ci > 0 and epoch % self.ci == 0:
            self.recorder.checkpoints[epoch] = deepcopy(self.recorder.model.state_dict())
            s = True
        if s:
            self.save(epoch)