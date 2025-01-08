import copy
import os
import torch
import torch_neuronx
from diffusers import FluxPipeline
from functools import partial
from model import TracingCLIPTextEncoderWrapper

from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance

COMPILER_WORKDIR_ROOT = os.path.dirname(__file__)

ckpt_path = "/tmp/test_model_builder_ckpt.pt"
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"


def trace_text_encoder():
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16)
    text_encoder = copy.deepcopy(pipe.text_encoder)
    del pipe

    text_encoder_wrapped = TracingCLIPTextEncoderWrapper(text_encoder)
    torch.save(text_encoder_wrapped.state_dict(), ckpt_path)

    builder = ModelBuilder(router=None,
                           tp_degree=8,
                           checkpoint_loader=partial(torch.load, ckpt_path),
                           compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir'))

    emb = torch.zeros((1, 77), dtype=torch.int64)

    builder.add(key="clip",
                model_instance=BaseModelInstance(partial(TracingCLIPTextEncoderWrapper, text_encoder=text_encoder), input_output_aliases={}),
                example_inputs=[(emb,)],
                compiler_args="""--target=trn2 --model-type=unet-inference""")
    weights_path = os.path.join(COMPILER_WORKDIR_ROOT, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    builder.shard_checkpoint(os.path.join(weights_path, 'clip'))

    traced_model = builder.trace(initialize_model_weights=False)
    compiled_model_path = os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiled_model')
    if not os.path.exists(compiled_model_path):
        os.mkdir(compiled_model_path)
    torch.jit.save(traced_model, os.path.join(compiled_model_path, 'clip.pt'))

    del text_encoder


if __name__ == '__main__':
    trace_text_encoder()
