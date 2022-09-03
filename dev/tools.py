import torch

def get_parameter_number(model: torch.nn.Module, printable:bool=False) -> dict:
    """
    stat the total param num and the num of trainable
    model: the model to be evaluated.
    ret: the dict of "Total" and "Trainable"
    """
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_info = {'Total': total_num, 'Trainable': trainable_num}
    if printable:
        for key, value in model_info.items():
            print(key, value, sep="\t")
    return model_info