import os
import torch


def load_optimizer(args, model):
    scheduler = None
    if args.optimizer == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    else:
        raise NotImplementedError

    return optimizer, scheduler


def save_model(path, model):
    out = os.path.join(path, 'best_model.tar')
    torch.save(model.state_dict(), out)
    return out
