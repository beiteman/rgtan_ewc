import copy


class early_stopper(object):
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Initialize the early stopper
        :param patience: the maximum number of rounds tolerated
        :param verbose: whether to stop early
        :param delta: the regularization factor
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.best_value = None
        self.best_cv = None
        self.is_earlystop = False
        self.count = 0
        self.best_model = None
        # self.val_preds = []
        # self.val_logits = []

    def earlystop(self, loss, model=None):  # , preds, logits):
        """
        :param loss: the loss score on validation set
        :param model: the model
        """
        value = -loss
        cv = loss
        # value = ap

        if self.best_value is None:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to('cpu')
            # self.val_preds = preds
            # self.val_logits = logits
        elif value < self.best_value + self.delta:
            self.count += 1
            if self.verbose:
                print('EarlyStoper count: {:02d}'.format(self.count))
            if self.count >= self.patience:
                self.is_earlystop = True
        else:
            self.best_value = value
            self.best_cv = cv
            self.best_model = copy.deepcopy(model).to('cpu')
            # self.val_preds = preds
            # self.val_logits = logits
            self.count = 0

class EWC:
    def __init__(self, model, dataloader, loss_fn, device):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        fisher = {n: torch.zeros_like(p, device=self.device) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.eval()
        for input_nodes, seeds, blocks in dataloader:
            self.model.zero_grad()
            batch_inputs, batch_work_inputs, batch_neighstat_inputs, batch_labels, lpa_labels = load_lpa_subtensor(
                num_feat, cat_feat, nei_feat, {}, labels, seeds, input_nodes, self.device, blocks
            )
            blocks = [block.to(self.device) for block in blocks]
            logits = self.model(blocks, batch_inputs, lpa_labels, batch_work_inputs, batch_neighstat_inputs)
            loss = self.loss_fn(logits, batch_labels[batch_labels != 2])
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
        return fisher

    def penalty(self, model):
        loss = 0
        for n, p in model.named_parameters():
            if n in self.params:
                _loss = self.fisher[n] * (p - self.params[n]) ** 2
                loss += _loss.sum()
        return loss

