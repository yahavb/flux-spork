import argparse
import copy
import os
import torch
import torch_neuronx
from diffusers import FluxPipeline
from functools import partial
from model import TracingVAEDecoderWrapper

from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance

COMPILER_WORKDIR_ROOT = os.path.dirname(__file__)
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"

ckpt_path = "/tmp/test_model_builder_ckpt.pt"


def trace_vae(height, width):
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.float32)
    decoder = copy.deepcopy(pipe.vae.decoder)
    del pipe

    decoder_wrapped = TracingVAEDecoderWrapper(decoder)
    torch.save(decoder_wrapped.state_dict(), ckpt_path)

    builder = ModelBuilder(router=None,
                           tp_degree=4,
                           checkpoint_loader=partial(torch.load, ckpt_path),
                           compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir'))

    latents = torch.rand([1, 16, height // 8, width // 8],
                         dtype=torch.bfloat16)

    builder.add(key="decoder",
                model_instance=BaseModelInstance(partial(TracingVAEDecoderWrapper, decoder=decoder), input_output_aliases={}),
                example_inputs=[(latents,)],
                compiler_args="""--target=trn2 --model-type=unet-inference""")
    weights_path = os.path.join(COMPILER_WORKDIR_ROOT, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    builder.shard_checkpoint(os.path.join(weights_path, 'decoder'))

    traced_model = builder.trace(initialize_model_weights=False)
    compiled_model_path = os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiled_model')
    if not os.path.exists(compiled_model_path):
        os.mkdir(compiled_model_path)
    torch.jit.save(traced_model, os.path.join(compiled_model_path, 'decoder.pt'))

    del decoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    trace_vae(args.height, args.width)