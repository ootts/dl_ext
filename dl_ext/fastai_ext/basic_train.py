from typing import Optional, List

import torch
from fastai.basic_train import validate, NoneReduceOnCPU
from fastai.callback import CallbackHandler
from fastai.core import PBar
from fastai.torch_core import OptLossFunc
from torch import nn
from torch.utils.data import DataLoader


def get_preds(model: nn.Module, dl: DataLoader, pbar: Optional[PBar] = None,
              cb_handler: Optional[CallbackHandler] = None,
              activ: nn.Module = None, loss_func: OptLossFunc = None,
              n_batch: Optional[int] = None) -> List[torch.Tensor]:
    "Tuple of predictions and targets, and optional losses (if `loss_func`) using `dl`, max batches `n_batch`."
    res = [torch.cat(o).cpu() for o in
           zip(*validate(model, dl, loss_func, cb_handler=cb_handler, pbar=pbar, average=False, n_batch=n_batch))]
    if loss_func is not None:
        with NoneReduceOnCPU(loss_func) as lf: res.append(lf(res[0], res[1]))
    if activ is not None: res[0] = activ(res[0])
    return res
