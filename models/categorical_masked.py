import torch
from torch.distributions.categorical import Categorical


class CategoricalMasked(Categorical):
    def __init__(self, logits=None, masks=None, device="cpu"):
        self.masks = masks
        if masks is not None and masks.any():
            logits = torch.where(masks, logits, torch.tensor(-1e8).to(device))
        super().__init__(logits=logits)

    def entropy(self):
        if self.masks is None or not self.masks.any():
            return super().entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.0).to(p_log_p.device))
        return -p_log_p.sum(-1)
