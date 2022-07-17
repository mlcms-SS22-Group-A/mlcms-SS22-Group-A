import torch
import pytorch_lightning as pl


class Euler(pl.LightningModule):
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
        return x + self.hparams["delta_t"] * self.model(x)

    def training_step(self, batch):
        pos, pos_target = batch

        pred = self.forward(pos.float())

        loss_fn = torch.nn.MSELoss()
        loss = loss_fn(pred, pos_target.float())
        self.log("loss", loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), self.hparams["learning_rate"])

