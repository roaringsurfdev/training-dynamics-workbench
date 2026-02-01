import os
from pathlib import Path

import copy

import numpy as np

import torch
from transformer_lens import HookedTransformerConfig, HookedTransformer
import einops

import tqdm.auto as tqdm


class ModuloAdditionSpecification:
    """Baseline modulo addition model"""

    def __init__(self, model_dir, prime, device, seed=999, data_seed=598, training_fraction=0.3):
        self.model = None
        self.name = "modulo_addition"
        self.model_dir = model_dir
        self.full_dir = os.path.join(model_dir, self.name)
        self.full_path = os.path.join(self.model_dir, self.name, f"{self.name}.pth")
        os.makedirs(Path(self.full_path).parent, exist_ok=True)
        self.prime = prime
        self.device = device
        self.seed = seed
        self.data_seed = data_seed
        self.training_fraction = training_fraction
        self.train_losses = []
        self.test_losses = []

    def create_model(self):
        # %% model definition
        cfg = HookedTransformerConfig(
            n_layers = 1,
            n_heads = 4,
            d_model = 128,
            d_head = 32,
            d_mlp = 512,
            act_fn = "relu",
            normalization_type=None,
            d_vocab=self.prime+1,
            d_vocab_out=self.prime,
            n_ctx=3,
            init_weights=True,
            device=self.device,
            seed = self.seed,
        )
        model = HookedTransformer(cfg)

        # Disable the biases, as we don't need them for this task and it makes things easier to interpret.
        for name, param in model.named_parameters():
            if "b_" in name:
                param.requires_grad = False
        
        self.model = model

        return self.model
    
    def load_from_file(self) -> HookedTransformer:
        # load the model
        model = self.create_model()
        cached_data = torch.load(self.full_path, weights_only=False)
        model.load_state_dict(cached_data['model'])
        self.model_checkpoints = cached_data["checkpoints"]
        self.checkpoint_epochs = cached_data["checkpoint_epochs"]
        self.test_losses = cached_data['test_losses']
        self.train_losses = cached_data['train_losses']
        self.train_indices = cached_data["train_indices"]
        self.test_indices = cached_data["test_indices"]

        self.model = model

        return self.model
    
    def loss_function(self, logits, labels):
        if len(logits.shape)==3:
            logits = logits[:, -1]
        logits = logits.to(torch.float64)
        log_probs = logits.log_softmax(dim=-1)
        correct_log_probs = log_probs.gather(dim=-1, index=labels[:, None])[:, 0]
        return -correct_log_probs.mean()
    
    def get_optimizer(self, model):
        # optimizer config
        lr = 1e-3
        wd = 1.
        betas = (0.9, 0.98)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, betas=betas)
        return optimizer

    
    def train(self) -> HookedTransformer:

        train_data, train_labels, test_data, test_labels, train_indices, test_indices = self.generate_training_data()
        
        num_epochs = 25000
        checkpoint_every = 100
        train_losses = []
        test_losses = []
        model_checkpoints = []
        checkpoint_epochs = []

        model = self.create_model()
        optimizer = self.get_optimizer(model)

        for epoch in tqdm.tqdm(range(num_epochs)):
            train_logits = model(train_data)
            train_loss = self.loss_function(train_logits, train_labels)
            train_loss.backward()
            train_losses.append(train_loss.item())

            optimizer.step()
            optimizer.zero_grad()

            with torch.inference_mode():
                test_logits = model(test_data)
                test_loss = self.loss_function(test_logits, test_labels)
                test_losses.append(test_loss.item())
            
            if ((epoch+1)%checkpoint_every)==0:
                checkpoint_epochs.append(epoch)
                model_checkpoints.append(copy.deepcopy(model.state_dict()))
                print(f"Epoch {epoch} Train Loss {train_loss.item()} Test Loss {test_loss.item()}")

        # save the model outputs
        torch.save(
            {
                "model":model.state_dict(),
                "config": model.cfg,
                "checkpoints": model_checkpoints,
                "checkpoint_epochs": checkpoint_epochs,
                "test_losses": test_losses,
                "train_losses": train_losses,
                "train_indices": train_indices,
                "test_indices": test_indices,
            },
            self.full_path)
        
        self.test_losses = test_losses
        self.train_losses = train_losses
        self.train_indices = train_indices
        self.test_indices = test_indices

        self.model = model

        return self.model
    
    def generate_training_data(self):
        a_vector = einops.repeat(torch.arange(self.prime), "i -> (i j)", j=self.prime)
        b_vector = einops.repeat(torch.arange(self.prime), "j -> (i j)", i=self.prime)
        equals_vector = einops.repeat(torch.tensor(self.prime), " -> (i j)", i=self.prime, j=self.prime)

        dataset = torch.stack([a_vector, b_vector, equals_vector], dim=1).to(self.device)
        self.dataset = dataset
        labels = (dataset[:, 0] + dataset[:, 1]) % self.prime

        torch.manual_seed(self.data_seed)
        indices = torch.randperm(self.prime*self.prime)
        cutoff = int(self.prime*self.prime*self.training_fraction)
        train_indices = indices[:cutoff]
        test_indices = indices[cutoff:]

        train_data = dataset[train_indices]
        train_labels = labels[train_indices]
        test_data = dataset[test_indices]
        test_labels = labels[test_indices]

        return train_data, train_labels, test_data, test_labels, train_indices, test_indices

    def compute_probe_activations(self, probe):
        probe_logits, probe_cache = self.model.run_with_cache(probe)
        return probe_logits, probe_cache

    def compute_uniform_loss(self):
        return np.log(self.prime)