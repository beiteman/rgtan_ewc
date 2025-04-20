
import torch
import copy

class EWC:
    def __init__(self, model: torch.nn.Module, dataloader, device, loss_fn, fisher_n_samples=500):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.dataloader = dataloader
        self.fisher_n_samples = fisher_n_samples

        self.params_old = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher()

    def _compute_fisher(self):
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()

        count = 0
        for input_nodes, output_nodes, blocks in self.dataloader:
            if count >= self.fisher_n_samples:
                break
            self.model.zero_grad()
            output = self.model(blocks, input_nodes)  # 根據你實際 forward 寫法修改
            labels = blocks[-1].dstdata['label']
            loss = self.loss_fn(output, labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.grad is not None and n in fisher:
                    fisher[n] += p.grad.detach() ** 2

            count += 1

        for n in fisher:
            fisher[n] /= count

        return fisher

    def compute_ewc_loss(self, model: torch.nn.Module):
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                _loss = self.fisher[n] * (p - self.params_old[n]) ** 2
                loss += _loss.sum()
        return loss
