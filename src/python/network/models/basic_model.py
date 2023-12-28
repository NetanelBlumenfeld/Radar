import torch
import torch.nn as nn


class BasicModel(nn.Module):
    def __init__(self, model_name: str):
        super(BasicModel, self).__init__()
        self.model_name = model_name

    @classmethod
    def load_model(
        cls,
        model_dir: str,
        optimizer_class,
        optimizer_args,
        device: torch.device,
        *args,
        **kwargs
    ):
        # Create an instance of the subclass with provided arguments
        model = cls(*args, **kwargs)
        model.to(device)

        # Load the checkpoint
        checkpoint = torch.load(model_dir, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        # Initialize the optimizer
        optimizer = optimizer_class(model.parameters(), **optimizer_args)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        # Other information from checkpoint
        epoch = checkpoint["epoch"]
        loss = checkpoint["loss"]

        return model, optimizer, epoch, loss
