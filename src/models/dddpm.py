import lightning as L
import torch
import torch.nn as nn

import src.backbone
from src.utils.vulpix_utils import (
    calc_diffusion_hyperparams2,
    get_mask,
    get_optimizer,
    masked_components_fft_amplitude,
    sampling_caldas2,
    training_loss_caldas,
    weighted_mse_loss,
)


class DDDPM(L.LightningModule):
    """decomposition-style diffusion process with selectable backbone."""

    def __init__(
        self,
        backbone_config=None,
        diffusion_config=None,
        model_config=None,
        lr=2e-4,
        ns_path = None,
        masking="forecast",
        max_components=4,
        monte_carlo=False,
        decomposition_method="fft",
        fixed_components=True,
        loss_name="mse",
        alpha=1e-9,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["kwargs"])

        if backbone_config is None:
            backbone_config = model_config
        if backbone_config is None:
            raise ValueError("backbone_config (or model_config) must be provided.")
        if diffusion_config is None:
            raise ValueError("diffusion_config must be provided.")


        self.monte_carlo = diffusion_config.get("monte_carlo", monte_carlo)
        self.decomposition_method = diffusion_config.get("decomposition_method", decomposition_method)
        self.fixed_components = diffusion_config.get("fixed_components", fixed_components)
        self.loss_name = diffusion_config.get("loss_name", loss_name)
        self.max_components = diffusion_config.get("max_components", max_components)
        self.masking = diffusion_config.get("masking", masking)
        bb_conf = dict(backbone_config)
        bb_name = bb_conf.pop("name", "DiffWaveImputer")
        bb_class = getattr(src.backbone, bb_name)
        self.net = bb_class(**bb_conf)
        self.lr = lr
        self.alpha = alpha
        #diffusion_hyperparams = calc_diffusion_hyperparams2(**diffusion_config)
        diffusion_hyperparams = torch.load(ns_path)
        self.register_buffer("Beta", diffusion_hyperparams["betas"].float())
        self.register_buffer("Alpha", diffusion_hyperparams["alphas"].float())
        self.register_buffer("Alpha_bar", diffusion_hyperparams["alpha_bars"].float())
        self.register_buffer("Sigma", diffusion_hyperparams["sigma"].float())
        self.dh = {'Beta': self.Beta, 'Alpha': self.Alpha, 'Alpha_bar': self.Alpha_bar, 'Sigma': self.Sigma, 'T': diffusion_config.get("T", 200) // diffusion_config.get("max_components", 1)}
        if loss_name == "mse":
            self.loss_fn = nn.MSELoss()
        elif loss_name == "wmse":
            self.loss_fn = weighted_mse_loss
        else:
            raise ValueError(f"Unsupported loss_name: {loss_name}")


    def _prepare_batch(self, batch):
        x_bt = batch["x"]
        #mask, loss_mask = get_mask(
        #    x_bt,
        #    self.hparams.masking,
        #    self.hparams.missing_k,
        #) # masking is useless, it will be removed TODO
        audio = x_bt.permute(0, 2, 1)
        cond = batch['c']
        cond = cond.permute(0, 2, 1) # sample, channel, time-steps
        mask = None
        loss_mask = None
        return audio, cond, mask, loss_mask

    def training_step(self, batch, batch_idx):
        _ = batch_idx
        audio, cond, mask, loss_mask = self._prepare_batch(batch)

        x = (audio, cond, mask, loss_mask)
        loss = training_loss_caldas(
            net=self.net,
            loss_fn=self.loss_fn,
            x=x,
            diffusion_hyperparams=self.dh,
            max_components=self.hparams.max_components,
            only_generate_missing=self.hparams.only_generate_missing,
            monte_carlo=self.hparams.monte_carlo,
            decomposition_method=self.hparams.decomposition_method,
        )

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        _ = batch_idx
        audio, cond, mask, loss_mask = self._prepare_batch(batch)

        x = (audio, cond, mask, loss_mask)
        val_loss = training_loss_caldas(
            net=self.net,
            loss_fn=nn.MSELoss(),
            x=x,
            diffusion_hyperparams=self.dh,
            max_components=self.max_components,
            only_generate_missing=self.hparams.only_generate_missing,
            monte_carlo=self.hparams.monte_carlo,
            decomposition_method=self.hparams.decomposition_method,
        )
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.hparams.eval_sampling:
            if self.hparams.max_components > 0:
                dk_val = masked_components_fft_amplitude(
                    audio,
                    mask,
                    self.hparams.max_components,
                    self.hparams.masking,
                    fixed=self.hparams.fixed_components,
                    decomposition_method=self.hparams.decomposition_method,
                )
            else:
                dk_val = None

            generated, _ = sampling_caldas2(
                self.net,
                audio.shape,
                self._build_diffusion_hyperparams(),
                cond=audio,
                mask=mask,
                max_components=self.hparams.max_components,
                dk=dk_val,
                only_generate_missing=self.hparams.only_generate_missing,
                guidance_weight=self.hparams.guidance_weight,
                sampling_with_dk=0,
                max_components_gen=self.hparams.max_components + 1,
            )
            diffusion_mse = nn.functional.mse_loss(generated[~mask.bool()], audio[~mask.bool()])
            self.log(
                "val_diffusion_loss",
                diffusion_mse,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def predict_step(self, batch, batch_idx):
        _ = batch_idx
        audio, cond, mask, _ = self._prepare_batch(batch)

        if self.hparams.max_components > 0:
            dk = masked_components_fft_amplitude(
                cond,
                mask,
                self.hparams.max_components,
                self.hparams.masking,
                fixed=self.hparams.fixed_components,
                decomposition_method=self.hparams.decomposition_method,
            )
        else:
            dk = None

        generated, _ = sampling_caldas2(
            self.net,
            audio.shape,
            self._build_diffusion_hyperparams(),
            cond=cond,
            mask=mask,
            max_components=self.hparams.max_components,
            dk=dk,
            only_generate_missing=self.hparams.only_generate_missing,
            guidance_weight=self.hparams.guidance_weight,
            sampling_with_dk=0,
            max_components_gen=self.hparams.max_components + 1,
        )
        return generated.permute(0, 2, 1)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.alpha)
