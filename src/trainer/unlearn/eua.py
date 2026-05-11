from trainer.unlearn.grad_diff import GradDiff
import torch
import torch.nn.functional as F
from trainer.utils import compute_eua_loss

class EUA(GradDiff):
    def __init__(
        self, beta1=0.3, beta2=1.0, gamma=1.0, alpha=0.1, *args, **kwargs
    ):  # attention, satimp requires two beta!!!!
        super().__init__(*args, **kwargs)
        self.beta1 = beta1
        self.beta2 = beta2
        self.gamma = gamma
        self.alpha = alpha
        if self.ref_model is None:
            self.ref_model = self._prepare_ref_model(self.model)
    
    def compute_loss(self, model, inputs, return_outputs=False):
        forget_inputs = inputs["forget"]
        forget_inputs = {
            "input_ids": forget_inputs["input_ids"],
            "attention_mask": forget_inputs["attention_mask"],
            "labels": forget_inputs["labels"],
        }

        retain_inputs = inputs["retain"]
        retain_inputs = {
            "input_ids": retain_inputs["input_ids"],
            "attention_mask": retain_inputs["attention_mask"],
            "labels": retain_inputs["labels"],
        }
        retain_loss = self.compute_retain_loss(model=model, retain_inputs=retain_inputs)
        eua_loss, outputs = compute_eua_loss(model=model, forget_inputs=forget_inputs, retain_inputs=retain_inputs, beta1=self.beta1, beta2=self.beta2, ref_model=self.ref_model)
        loss = self.gamma * eua_loss + self.alpha * retain_loss

        return (loss, outputs) if return_outputs else loss
