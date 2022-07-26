import torch
import pytorch_lightning as pl


class RungeKutta(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3, hparams["hidden_layer_1"]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hparams["hidden_layer_1"], hparams["hidden_layer_2"]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hparams["hidden_layer_2"], hparams["hidden_layer_3"]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hparams["hidden_layer_3"], 2)
        ).float()

    def forward(self, x):
        alpha = torch.reshape(x[:, 2], (x.shape[0], 1))
        delta_t = self.hparams["delta_t"]
        k0 = self.model(x)
        k1 = self.model(torch.cat((x[:, :2] + 0.5 * delta_t * k0, alpha), 1))
        k2 = self.model(torch.cat((x[:, :2] + 0.5 * delta_t * k1, alpha), 1))
        k3 = self.model(torch.cat((x[:, :2] + delta_t * k2, alpha), 1))
        return (1/6) * (k0 + 2 * k1 + 2 * k2 + k3)

    def training_step(self, batch):
        pos = batch[:, :3]
        pos_target = batch[:, 3:]

        pred = pos.float()[:, :2] + self.hparams["delta_t"] * self.forward(pos.float())

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(pred, pos_target.float())
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), self.hparams["learning_rate"])

