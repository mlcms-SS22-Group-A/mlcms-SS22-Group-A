import torch
import pytorch_lightning as pl


class Euler(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 32),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(32, 2),
            torch.nn.LeakyReLU(),
        )

    def forward(self, x):
        return x + self.hparams["delta_t"] * self.model(x)

    def training_step(self, batch):
        x, x_target = batch

        pred = self.forward(x)

        loss_fn = torch.nn.MSELoss()
        return loss_fn(pred, x_target)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters())
