import torch
import torch.nn as nn

from transformers.modeling_outputs import BaseModelOutput
from typing import Optional, Union, Tuple


class TracingT5TextEncoderWrapper(nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.neuron_text_encoder = text_encoder
        self.config = text_encoder.config
        self.dtype = text_encoder.dtype
        self.device = text_encoder.device

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = False,
    ) -> Union[Tuple[torch.FloatTensor], BaseModelOutput]:
        return_dict = return_dict if return_dict is not None \
            else self.config.use_return_dict

        encoder_outputs = self.neuron_text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        return encoder_outputs
