import gc
from abc import ABC, abstractmethod

import torch
from transformers import Trainer, TrainingArguments

import wandb
from locket.training.circuit_breaking.utils import (
    forward_with_cache_for_modules,
    unfreeze_specific_layers,
)


def apply_short_circuiting(
    model,
    data_loader,
    eval_func: callable,
    eval_kwargs: dict,
    grad_accum_steps: int,
    eval_every: int,
    param_filter=lambda param: True,
    rr_target_layers: list | None = None,
    retain_target_layers: list | None = None,
    lr=1e-5,
    num_epochs=1,
    alpha=5.0,
    alpha_rr_min=0.125,
    alpha_retain_max=4.825,
    k=32,
    main_device: str | torch.device = "cuda",
):
    model = model.train()

    if retain_target_layers is None:
        retain_target_layers = range(model.model.config.num_hidden_layers)
    if rr_target_layers is None:
        rr_target_layers = [10, 20]

    unfreeze_specific_layers(model, param_filter)
    # apply_gradient_checkpointing_to_unfrozen_layers(model)

    alpha_scheduler = AlphaScheduler(
        alpha=alpha,
        T=len(data_loader),
        alpha_rr_min=alpha_rr_min,
        alpha_retain_max=alpha_retain_max,
        k=k,
    )

    training_args = TrainingArguments(
        output_dir="./models",  # Choose an appropriate output directory
        learning_rate=lr,
        num_train_epochs=num_epochs,
        gradient_accumulation_steps=grad_accum_steps,
        eval_strategy="steps",
        eval_steps=eval_every,
        report_to=["wandb"],
        do_eval=True,
        do_train=True,
        remove_unused_columns=False,
        warmup_steps=0,
    )

    trainer = ShortCircuitTrainer(
        model=model,
        args=training_args,
        alpha_scheduler=alpha_scheduler,
        eval_fn=eval_func,
        eval_kwargs=eval_kwargs,
        rr_target_layers=rr_target_layers,
        retain_target_layers=retain_target_layers,
        train_dataloader=data_loader,
        main_device=main_device,
    )

    # evaluate once before training starts to get a baseline
    results = trainer.evaluate()
    wandb.log(results)

    trainer.train()

    # evaluate once after training ends
    results = trainer.evaluate()
    wandb.log(results)


class AlphaScheduler:
    def __init__(self, alpha, T, alpha_rr_min=0, alpha_retain_max=0, k=1):
        self.alpha = alpha
        self.alpha_rr_min = alpha_rr_min
        self.alpha_retain_max = alpha_retain_max if alpha_retain_max > 0 else alpha
        self.k = k
        self.T = T
        self.t = 0

    def get_alpha_coeff(self):
        a_rr = self.alpha * (1 - self.k * self.t / (2 * self.T))
        a_rr = max(a_rr, self.alpha_rr_min)
        a_retain = self.alpha * self.k * self.t / (2 * self.T)
        a_retain = min(a_retain, self.alpha_retain_max)
        return a_rr, a_retain

    def step(self):
        self.t += 1


class RRTrainer(ABC, Trainer):
    def __init__(self, *args, **kwargs):
        self.alpha_scheduler = kwargs.pop("alpha_scheduler")
        self.eval_fn = kwargs.pop("eval_fn")
        self.eval_kwargs = kwargs.pop("eval_kwargs")
        self.rr_target_layers = kwargs.pop("rr_target_layers")
        self.retain_target_layers = kwargs.pop(
            "retain_target_layers",
            list(range(kwargs["model"].config.num_hidden_layers)),
        )
        self.custom_train_dataloader = kwargs.pop("train_dataloader", None)
        self.main_device = kwargs.pop("main_device", "cuda")
        self.rr_modules = torch.nn.ModuleList(
            [kwargs["model"].model.model.layers[i] for i in self.rr_target_layers]
        )
        self.retain_modules = torch.nn.ModuleList(
            [kwargs["model"].model.model.layers[i] for i in self.retain_target_layers]
        )
        self.batch_loss = 0.0
        self.batch_loss_rr = 0.0
        self.batch_loss_retain = 0.0
        self.steps_since_last_log = 0
        super().__init__(*args, **kwargs)

    def stack_tensors_on_device(self, tensors):
        """
        Stacks a list of tensors ensuring they are on the same device.

        Parameters:
        - tensors (list of torch.Tensor): List of tensors to stack.
        - device (torch.device or str, optional): The device to move tensors to before stacking. If None,
        the device of the first tensor is used.

        Returns:
        - torch.Tensor: Stacked tensor on the specified device.
        """
        if not tensors:
            raise ValueError("The list of tensors should not be empty")
        if self.main_device == "cuda":
            return torch.stack(tensors)

        tensors_on_device = [tensor.to(self.main_device) for tensor in tensors]
        return torch.stack(tensors_on_device)

    def get_train_dataloader(self):
        if self.custom_train_dataloader is not None:
            return self.custom_train_dataloader
        return super().get_train_dataloader()

    def get_retain_hidden_states(
        self, model, inputs, target_positions=None, no_grad=True
    ):
        outputs = forward_with_cache_for_modules(
            model,
            {"input_ids": inputs["retain_ids"]},
            self.retain_modules,
            self.retain_target_layers,
            target_positions=target_positions,
            no_grad=no_grad,
        )
        return [outputs[layer] for layer in self.retain_target_layers]

    def get_rr_hidden_states(self, model, inputs, target_positions=None, no_grad=True):
        outputs = forward_with_cache_for_modules(
            model,
            {"input_ids": inputs["rr_ids"]},
            self.rr_modules,
            self.rr_target_layers,
            target_positions=target_positions,
            no_grad=no_grad,
        )
        return [outputs[layer] for layer in self.rr_target_layers]

    @abstractmethod
    def get_rr_hidden_states_with_intervention(
        self, model, input_ids, target_positions=None, no_grad=True
    ):
        pass

    @abstractmethod
    def get_retain_loss(self, updated_hidden, frozen_hidden, attention_mask):
        pass

    @abstractmethod
    def get_rerouting_loss(self, updated_hidden, frozen_hidden, attention_mask):
        pass

    def compute_loss(self, model, inputs, return_outputs=False):
        # Repeat the attention masks for each target layer
        # Afterwards, attention masks have shape (n_layers, batch_size, seq, 1)
        retain_attention_mask = inputs["retain_attention_mask"]
        rr_attention_mask = inputs["rr_attention_mask"]
        rr_attention_mask = rr_attention_mask.repeat(
            len(self.rr_target_layers), 1, 1
        ).unsqueeze(-1)
        retain_attention_mask = retain_attention_mask.repeat(
            len(self.retain_target_layers), 1, 1
        ).unsqueeze(-1)

        # Only keep the attention masks for the prediction tokens
        # Afterwards, attention masks have shape (n_layers, batch_size, # prediction tokens, 1)
        retain_prediction_idxs = inputs["retain_prediction_idxs"]
        rr_prediction_idxs = inputs["rr_prediction_idxs"]
        rr_attention_mask = rr_attention_mask[:, :, rr_prediction_idxs]
        retain_attention_mask = retain_attention_mask[:, :, retain_prediction_idxs]

        if not self.main_device == "cuda" and not self.main_device == "cpu":
            retain_prediction_idxs = retain_prediction_idxs.to(self.main_device)
            rr_prediction_idxs = rr_prediction_idxs.to(self.main_device)

        self.alpha_scheduler.step()
        a_rr, a_retain = self.alpha_scheduler.get_alpha_coeff()

        # Get representations from the frozen model
        model.eval()
        with model.disable_adapter():
            with torch.no_grad():
                if a_retain > 0:
                    frozen_retain_hidden = self.get_retain_hidden_states(
                        model, inputs, target_positions=retain_prediction_idxs
                    )
                    frozen_retain_hidden = self.stack_tensors_on_device(
                        frozen_retain_hidden
                    )
                    # frozen_retain_hidden = frozen_retain_hidden[:, :, retain_prediction_idxs]  # (n_layers, batch_size, # prediction tokens, hidden_size)
                if a_rr > 0:
                    frozen_rr_hidden = self.get_rr_hidden_states_with_intervention(
                        model, inputs, target_positions=rr_prediction_idxs
                    )
                    frozen_rr_hidden = self.stack_tensors_on_device(frozen_rr_hidden)
                    # frozen_rr_hidden = frozen_rr_hidden[:, :, rr_prediction_idxs]  # (n_layers, batch_size, # prediction tokens, hidden_size)

        # Get representations from the updated model and calculate loss
        model.train()
        retain_loss = torch.tensor(0.0, device=self.main_device)
        rr_loss = torch.tensor(0.0, device=self.main_device)
        if a_retain > 0:
            updated_retain_hidden = self.get_retain_hidden_states(
                model, inputs, target_positions=retain_prediction_idxs, no_grad=False
            )
            updated_retain_hidden = self.stack_tensors_on_device(updated_retain_hidden)
            # updated_retain_hidden = updated_retain_hidden[:, :, retain_prediction_idxs]  # (n_layers, batch_size, # prediction tokens, hidden_size)
            retain_loss = self.get_retain_loss(
                updated_retain_hidden, frozen_retain_hidden, retain_attention_mask
            )

        if a_rr > 0:
            updated_rr_hidden = self.get_rr_hidden_states(
                model, inputs, target_positions=rr_prediction_idxs, no_grad=False
            )
            updated_rr_hidden = self.stack_tensors_on_device(updated_rr_hidden)
            # updated_rr_hidden = updated_rr_hidden[:, :, rr_prediction_idxs]  # (n_layers, batch_size, # prediction tokens, hidden_size)
            rr_loss = self.get_rerouting_loss(
                updated_rr_hidden, frozen_rr_hidden, rr_attention_mask
            )

        loss = a_retain * retain_loss + a_rr * rr_loss

        # Accumulate the loss
        self.batch_loss += loss.item()
        self.batch_loss_rr += rr_loss.item()
        self.batch_loss_retain += retain_loss.item()
        self.steps_since_last_log += 1

        # Check if this is an optimization step
        if self.steps_since_last_log % self.args.gradient_accumulation_steps == 0:
            # Log loss and alpha
            self.log({"loss": self.batch_loss})
            self.log({"loss_retain": self.batch_loss_retain})
            self.log({"loss_rr": self.batch_loss_rr})
            self.log({"alpha_rr": a_rr})
            self.log({"alpha_retain": a_retain})
            self.log({"lr": self.optimizer.param_groups[0]["lr"]})

            # Reset the cumulative loss and step counter
            self.batch_loss = 0.0
            self.batch_loss_rr = 0.0
            self.batch_loss_retain = 0.0
            self.steps_since_last_log = 0

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        return (loss,) if return_outputs else loss

    # def evaluation_loop(self, dataloader, description, **kwargs):
    #     self.eval_fn(**self.eval_kwargs)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        return self.eval_fn(**self.eval_kwargs)

    def get_eval_dataloader(self, eval_dataset=None):
        return [None]


class ShortCircuitTrainer(RRTrainer):
    def get_rr_hidden_states_with_intervention(
        self, model, inputs, target_positions=None, no_grad=True
    ):
        return self.get_rr_hidden_states(
            model, inputs, target_positions=target_positions, no_grad=no_grad
        )

    def get_retain_loss(self, updated_hidden, frozen_hidden, attention_mask):
        frozen_hidden *= attention_mask
        updated_hidden *= attention_mask
        return torch.nn.functional.mse_loss(frozen_hidden, updated_hidden)

    def get_rerouting_loss(self, updated_hidden, frozen_hidden, attention_mask):
        updated_hidden *= attention_mask
        frozen_hidden *= attention_mask
        normalized_updated_rr_outputs = updated_hidden / (
            torch.norm(updated_hidden, dim=-1, keepdim=True, dtype=torch.float)
        )
        normalized_frozen_rr_outputs = frozen_hidden / (
            torch.norm(frozen_hidden, dim=-1, keepdim=True, dtype=torch.float)
        )
        return torch.mean(
            torch.nn.functional.relu(
                torch.nn.functional.cosine_similarity(
                    normalized_updated_rr_outputs, normalized_frozen_rr_outputs, dim=-1
                )
            )
        )


class PromptInterventionRRTrainer(RRTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_rr_hidden_states_with_intervention(
        self, model, inputs, target_positions=None, no_grad=True
    ):
        return forward_with_cache_for_modules(
            model,
            {"input_ids": inputs["rr_ids_with_intervention_prompt"]},
            self.rr_modules,
            self.rr_target_layers,
            target_positions=target_positions,
            no_grad=no_grad,
        )

    def get_retain_loss(self, updated_hidden, frozen_hidden, attention_mask):
        frozen_hidden *= attention_mask
        updated_hidden *= attention_mask
        return torch.nn.functional.mse_loss(updated_hidden, frozen_hidden)

    def get_rerouting_loss(self, updated_hidden, frozen_hidden, attention_mask):
        # updated_hidden has shape (n_layers, batch_size, k + seq, hidden_size) where k is the token lenght of the intervention prompt
        # frozen_hidden has shape (n_layers, batch_size, seq, hidden_size)
        # attention_mask has shape (n_layers, batch_size, seq)
        # we want to only consider the hidden states for the sequence tokens and not for the intervention prompt
        updated_hidden_seq = updated_hidden[:, :, -frozen_hidden.size(2) :, :]
        frozen_hidden *= attention_mask
        updated_hidden_seq *= attention_mask
        return torch.nn.functional.mse_loss(updated_hidden_seq, frozen_hidden)


class SteeringInterventionRRTrainer(RRTrainer):
    def __init__(self, *args, **kwargs):
        self.steering_vector = kwargs.pop("steering_vector")
        self.steering_multiplier = kwargs.pop("steering_multiplier")
        self.masking_fn = kwargs.pop("masking_fn")
        super().__init__(*args, **kwargs)

    def get_rr_hidden_states_with_intervention(
        self, model, inputs, target_positions=None, no_grad=True
    ):
        steering_mask = self.masking_fn(inputs["rr_ids"])

        with self.steering_vector.apply(
            model,
            multiplier=self.steering_multiplier,
            token_indices=steering_mask,
            min_token_index=None,
        ):
            rr_hidden_states = self.get_rr_hidden_states(
                model, inputs, target_positions=target_positions, no_grad=no_grad
            )

        return rr_hidden_states

    def get_retain_loss(self, updated_hidden, frozen_hidden, attention_mask):
        frozen_hidden *= attention_mask
        updated_hidden *= attention_mask
        return torch.nn.functional.mse_loss(updated_hidden, frozen_hidden)

    def get_rerouting_loss(self, updated_hidden, frozen_hidden, attention_mask):
        frozen_hidden *= attention_mask
        updated_hidden *= attention_mask
        return torch.nn.functional.mse_loss(updated_hidden, frozen_hidden)
