import torch
import pytorch_lightning as pl


class Euler(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(hparams["input_layer"], hparams["hidden_layer_1"]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hparams["hidden_layer_1"], hparams["hidden_layer_2"]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hparams["hidden_layer_2"], hparams["hidden_layer_3"]),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hparams["hidden_layer_3"], hparams["output_layer"]),
        )

    def forward(self, x):
        return x + self.hparams["delta_t"] * self.model(x)

    def training_step(self, batch):
        print("BATCH SHAPE: ", batch.shape)
        traj, traj_shifted = batch[0]
        print("traj SHAPE: ", traj.shape)



        # traj = batch["traj"]
        # target = batch["traj_shifted"]

        pred = self.forward(traj).view(self.hparams["num_datapoints"],2)

        loss_fn = torch.nn.MSELoss()
        return loss_fn(pred, target)

    def configure_optimizers(self):
        optim = torch.optim.Adam(self.model.parameters(),
            self.hparams["learning_rate"],
            weight_decay=self.hparams["weight_decay"]
        )
        # StepLR = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[30], gamma=0.5)
        return optim

