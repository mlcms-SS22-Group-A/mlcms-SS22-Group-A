import torch
import pytorch_lightning as pl


class Euler(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.model = None

    def forward(self, x):
        return x + self.hparams["delta_t"] * self.model(x)

    def training_step(self, batch):
        x, y = batch

        pred = self.forward(x)

        loss_fn = torch.nn.MSELoss()
        return loss_fn(pred, y)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())
