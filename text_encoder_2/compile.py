import argparse
import copy
import os
import torch
import torch_neuronx
from diffusers import FluxPipeline
from functools import partial
from model import TracingT5TextEncoderWrapper
from safetensors.torch import save_model

from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance

COMPILER_WORKDIR_ROOT = os.path.dirname(__file__)
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"

ckpt_path = "/tmp/test_model_builder_ckpt.pt"


def trace_text_encoder_2(max_sequence_length):
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16)
    text_encoder_2 = copy.deepcopy(pipe.text_encoder_2)
    del pipe

    text_encoder_wrapped = TracingT5TextEncoderWrapper(text_encoder_2)
    torch.save(text_encoder_wrapped.state_dict(), ckpt_path)

    emb = torch.zeros((1, max_sequence_length), dtype=torch.int64)

    builder = ModelBuilder(router=None,
                           tp_degree=8,
                           checkpoint_loader=partial(torch.load, ckpt_path),
                           compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir'))
    
    state_dict = torch.load(ckpt_path)
    for key in ['neuron_text_encoder.shared.weight', 'neuron_text_encoder.encoder.embed_tokens.weight']:
        if key in state_dict:
            state_dict[key] = state_dict[key].clone()
    torch.save(state_dict, ckpt_path)

    builder.add(key="t5",
                model_instance=BaseModelInstance(partial(TracingT5TextEncoderWrapper, text_encoder=text_encoder_2), input_output_aliases={}),
                example_inputs=[(emb,)],
                compiler_args="""--target=trn1 --model-type=unet-inference""")

    traced_model = builder.trace(initialize_model_weights=False)
    compiled_model_path = os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiled_model')
    if not os.path.exists(compiled_model_path):
        os.mkdir(compiled_model_path)
    torch.jit.save(traced_model, os.path.join(compiled_model_path, 't5.pt'))
    weights_path = os.path.join(COMPILER_WORKDIR_ROOT, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    builder.shard_checkpoint(os.path.join(weights_path, 't5'))


    del text_encoder_2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--max_sequence_length",
        type=int,
        default=512,
        help="maximum sequence length for the text embeddings"
    )
    args = parser.parse_args()
    trace_text_encoder_2(args.max_sequence_length)
