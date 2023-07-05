from pytorch_lightning import LightningModule, LightningDataModule
from pytorch_lightning.utilities.cli import LightningCLI

# see https://github.com/PyTorchLightning/pytorch-lightning/issues/10349
import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

cli = LightningCLI(LightningModule, LightningDataModule, 
                   subclass_mode_model=True, subclass_mode_data=True,
                   save_config_callback=None)