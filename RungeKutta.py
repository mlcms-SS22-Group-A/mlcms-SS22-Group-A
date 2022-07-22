import torch
import pytorch_lightning as pl


class RungeKutta(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2, hparams["hidden_layer_1"]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hparams["hidden_layer_1"], hparams["hidden_layer_2"]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hparams["hidden_layer_2"], hparams["hidden_layer_3"]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hparams["hidden_layer_3"], 2)
        ).float()

    def forward(self, x):
        delta_t = self.hparams["delta_t"]
        k0 = self.model(x)
        k1 = self.model(x + 0.5 * delta_t * k0)
        k2 = self.model(x + 0.5 * delta_t * k1)
        k3 = self.model(x + delta_t * k2)
        return x + (1/6) * delta_t * (k0 + 2 * k1 + 2 * k2 + k3)

    def training_step(self, batch):
        pos = batch[:, 0]
        pos_target = batch[:, 1]

        pred = self.forward(pos.float())

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(pred, pos_target.float())
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), self.hparams["learning_rate"])

