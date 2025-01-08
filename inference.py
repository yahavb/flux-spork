import argparse
import torch
import torch.nn as nn
import torch_neuronx
import neuronx_distributed
import os

from diffusers import FluxPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from safetensors.torch import load_file
from typing import Any, Dict, Optional, Union

COMPILER_WORKDIR_ROOT = os.path.dirname(__file__)
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"

TEXT_ENCODER_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'text_encoder_1/compiled_model/clip.pt')
TEXT_ENCODER_2_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'text_encoder_2/compiled_model/t5.pt')
VAE_DECODER_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'decoder/compiled_model/decoder.pt')

EMBEDDERS_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/embedders.pt')
OUT_LAYERS_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/out_layers.pt')
SINGLE_TRANSFORMER_BLOCKS_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/single_transformer_blocks.pt')
TRANSFORMER_BLOCKS_PATH = os.path.join(
    COMPILER_WORKDIR_ROOT,
    'transformer/compiled_model/transformer_blocks.pt')


class NeuronFluxTransformer2DModel(nn.Module):
    def __init__(
        self,
        config,
        x_embedder,
        context_embedder
    ):
        super().__init__()
        self.embedders_model = \
                torch.jit.load(EMBEDDERS_PATH)
        weights = []
        # start_rank_id = 0
        # local_ranks_size = 4
        # for rank in range(start_rank_id, start_rank_id + local_ranks_size):
        #     ckpt = load_file(
        #         os.path.join(
        #             COMPILER_WORKDIR_ROOT, f"transformer/weights/embedders/tp{rank}_sharded_checkpoint.safetensors"
        #         )
        #     )
        #     weights.append(ckpt)
        # start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
        self.embedders_model.nxd_model.initialize(weights)

        self.transformer_blocks_model = \
            torch.jit.load(
                TRANSFORMER_BLOCKS_PATH)
        weights = []
        # start_rank_id = 0
        # local_ranks_size = 4
        # for rank in range(start_rank_id, start_rank_id + local_ranks_size):
        #     ckpt = load_file(
        #         os.path.join(
        #             COMPILER_WORKDIR_ROOT, f"transformer/weights/transformer_blocks/tp{rank}_sharded_checkpoint.safetensors"
        #         )
        #     )
        #     weights.append(ckpt)
        # start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
        self.transformer_blocks_model.nxd_model.initialize(weights)

        self.single_transformer_blocks_model = \
            torch.jit.load(
                SINGLE_TRANSFORMER_BLOCKS_PATH)
        weights = []
        # start_rank_id = 0
        # local_ranks_size = 4
        # for rank in range(start_rank_id, start_rank_id + local_ranks_size):
        #     ckpt = load_file(
        #         os.path.join(
        #             COMPILER_WORKDIR_ROOT, f"transformer/weights/single_transformer_blocks/tp{rank}_sharded_checkpoint.safetensors"
        #         )
        #     )
        #     weights.append(ckpt)
        # start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
        self.single_transformer_blocks_model.nxd_model.initialize(weights)

        self.out_layers_model = \
            torch.jit.load(
                OUT_LAYERS_PATH)
        weights = []
        # start_rank_id = 0
        # local_ranks_size = 4
        # for rank in range(start_rank_id, start_rank_id + local_ranks_size):
        #     ckpt = load_file(
        #         os.path.join(
        #             COMPILER_WORKDIR_ROOT, f"transformer/weights/out_layers/tp{rank}_sharded_checkpoint.safetensors"
        #         )
        #     )
        #     weights.append(ckpt)
        # start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
        self.out_layers_model.nxd_model.initialize(weights)

        self.config = config
        self.x_embedder = x_embedder
        self.context_embedder = context_embedder
        self.device = torch.device("cpu")

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:

        hidden_states = self.x_embedder(hidden_states)

        hidden_states, temb, image_rotary_emb = self.embedders_model(
            hidden_states,
            timestep,
            guidance,
            pooled_projections,
            txt_ids,
            img_ids
        )

        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        image_rotary_emb = image_rotary_emb.type(torch.bfloat16)

        encoder_hidden_states, hidden_states = self.transformer_blocks_model(
            hidden_states,
            encoder_hidden_states,
            temb,
            image_rotary_emb
        )

        hidden_states = torch.cat([encoder_hidden_states, hidden_states],
                                  dim=1)

        hidden_states = self.single_transformer_blocks_model(
            hidden_states,
            temb,
            image_rotary_emb
        )

        hidden_states = hidden_states.to(torch.bfloat16)

        return self.out_layers_model(
            hidden_states,
            encoder_hidden_states,
            temb
        )


class NeuronFluxCLIPTextEncoderModel(nn.Module):
    def __init__(self, dtype, encoder):
        super().__init__()
        self.dtype = dtype
        self.encoder = encoder
        weights = []
        # start_rank_id = 0
        # local_ranks_size = 4
        # for rank in range(start_rank_id, start_rank_id + local_ranks_size):
        #     ckpt = load_file(
        #         os.path.join(
        #             COMPILER_WORKDIR_ROOT, f"text_encoder_1/weights/clip/tp{rank}_sharded_checkpoint.safetensors"
        #         )
        #     )
        #     weights.append(ckpt)
        # start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
        self.encoder.nxd_model.initialize(weights)
        self.device = torch.device("cpu")

    def forward(self, emb, output_hidden_states):
        output = self.encoder(emb)
        output = CLIPEncoderOutput(output)
        return output


class CLIPEncoderOutput():
    def __init__(self, dictionary):
        self.pooler_output = dictionary["pooler_output"]


class NeuronFluxT5TextEncoderModel(nn.Module):
    def __init__(self, dtype, encoder):
        super().__init__()
        self.dtype = dtype
        self.encoder = encoder
        weights = []
        self.encoder.nxd_model.initialize(weights)
        self.device = torch.device("cpu")

    def forward(self, emb, output_hidden_states):
        return torch.unsqueeze(self.encoder(emb)["last_hidden_state"], 1)


def run_inference(
        prompt,
        height,
        width,
        max_sequence_length,
        num_inference_steps):
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16)
    # with torch_neuronx.experimental.neuron_cores_context(start_nc=0):
    #     pipe.text_encoder = NeuronFluxCLIPTextEncoderModel(
    #         pipe.text_encoder.dtype,
    #         torch.jit.load(TEXT_ENCODER_PATH))
    # with torch_neuronx.experimental.neuron_cores_context(start_nc=8):
    #     pipe.text_encoder_2 = NeuronFluxT5TextEncoderModel(
    #         pipe.text_encoder_2.dtype,
    #         torch.jit.load(TEXT_ENCODER_2_PATH))
    pipe.transformer = NeuronFluxTransformer2DModel(
        pipe.transformer.config,
        pipe.transformer.x_embedder,
        pipe.transformer.context_embedder)
    with torch_neuronx.experimental.neuron_cores_context(start_nc=8):
        pipe.vae.decoder = torch.jit.load(VAE_DECODER_PATH)
        weights = []
        start_rank_id = 0
        local_ranks_size = 4
        for rank in range(start_rank_id, start_rank_id + local_ranks_size):
            ckpt = load_file(
                os.path.join(
                    COMPILER_WORKDIR_ROOT, f"decoder/weights/decoder/tp{rank}_sharded_checkpoint.safetensors"
                )
            )
            weights.append(ckpt)
        pipe.vae.decoder.nxd_model.initialize(weights)

    image = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=3.5,
        num_inference_steps=num_inference_steps,
        max_sequence_length=max_sequence_length
    ).images[0]
    image.save(os.path.join(COMPILER_WORKDIR_ROOT, "flux-dev.png"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--prompt",
        type=str,
        default="A cat holding a sign that says hello world",
        help="prompt for image to be generated; generates cat by default"
    )
    parser.add_argument(
        "-hh",
        "--height",
        type=int,
        default=1024,
        help="height of images to be generated by compilation of this model"
    )
    parser.add_argument(
        "-w",
        "--width",
        type=int,
        default=1024,
        help="width of images to be generated by compilation of this model"
    )
    parser.add_argument(
        "-m",
        "--max_sequence_length",
        type=int,
        default=512,
        help="maximum sequence length for the text embeddings"
    )
    parser.add_argument(
        "-n",
        "--num_inference_steps",
        type=int,
        default=50,
        help="number of inference steps to run in generating image"
    )
    args = parser.parse_args()
    run_inference(
        args.prompt,
        args.height,
        args.width,
        args.max_sequence_length,
        args.num_inference_steps)
