import torch
import torch.nn as nn


class TracingVAEDecoderWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(
        self,
        latents: torch.Tensor
    ):
        latents = latents.to(torch.float32)
        return self.decoder(
            latents
        )
