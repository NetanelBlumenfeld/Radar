import os

import torch
import torch.nn as nn


class BasicModel(nn.Module):
    def __init__(self, model_name: str, only_wights: bool = False):
        super(BasicModel, self).__init__()
        self.model_name = model_name
        self.only_wights = only_wights

    def save_model(self, save_dir: str):
        f_name = "wights_model.pth" if self.only_wights else "model.pt"
        path = os.path.join(self.save_dir, self.model_name, f_name)
        if self.only_wights:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self, path)
