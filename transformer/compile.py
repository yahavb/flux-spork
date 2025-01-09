import argparse
import copy
import model
import os
import shutil
import torch
import torch_xla.distributed.xla_multiprocessing as xmp
from functools import partial

from collections import OrderedDict
from diffusers import FluxPipeline
from neuronx_distributed.trace.model_builder import ModelBuilder, BaseModelInstance

COMPILER_WORKDIR_ROOT = os.path.dirname(__file__)
os.environ["NEURON_RT_VIRTUAL_CORE_SIZE"] = "2"

ckpt_path = "/tmp/test_model_builder_ckpt.pt"

def get_ckpt():
    ckpt = OrderedDict()
    return ckpt


def generate_transformer_model(transformer, height, width, max_sequence_length):
    generate_transformer_embedders_model(transformer, height, width, max_sequence_length)
    generate_transformer_blocks_model(transformer, height, width, max_sequence_length)
    generate_transformer_single_blocks_model(transformer, height, width, max_sequence_length)
    generate_transformer_out_layers_model(transformer, height, width, max_sequence_length)

def generate_transformer_embedders_model(transformer, height, width, max_sequence_length):
    embedders = model.TracingTransformerEmbedderWrapper(transformer.time_text_embed, transformer.pos_embed, False)
    torch.save(embedders.state_dict(), ckpt_path)
    builder = ModelBuilder(router=None,
                           tp_degree=8,
                           checkpoint_loader=get_ckpt,
                           compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir_embedders'))

    hidden_states = torch.rand([1, height * width // 256, 3072],
                               dtype=torch.bfloat16)
    timestep = torch.rand([1], dtype=torch.bfloat16)
    guidance = torch.rand([1], dtype=torch.float32)
    pooled_projections = torch.rand([1, 768], dtype=torch.bfloat16)
    txt_ids = torch.rand([1, max_sequence_length, 3], dtype=torch.bfloat16)
    img_ids = torch.rand([1, height * width // 256, 3], dtype=torch.bfloat16)
    sample_inputs = hidden_states, timestep, guidance, pooled_projections, \
        txt_ids, img_ids

    builder.add(key="embedders",
                model_instance=BaseModelInstance(partial(model.TracingTransformerEmbedderWrapper, time_text_embed=transformer.time_text_embed, pos_embed=transformer.pos_embed, is_distributed=True), input_output_aliases={}),
                example_inputs=[sample_inputs],
                compiler_args="""--target=trn2 --model-type=transformer --lnc=2 -O1 --tensorizer-options='--run-pg-layout-and-tiling'""")

    traced_model = builder.trace(initialize_model_weights=False)
    compiled_model_path = os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiled_model')
    if not os.path.exists(compiled_model_path):
        os.mkdir(compiled_model_path)
    torch.jit.save(traced_model, os.path.join(compiled_model_path, 'embedders.pt'))
    weights_path = os.path.join(COMPILER_WORKDIR_ROOT, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    builder.shard_checkpoint(os.path.join(weights_path, 'embedders'))


def generate_transformer_blocks_model(transformer, height, width, max_sequence_length):
    blocks = model.TracingTransformerBlockWrapper(transformer, transformer.transformer_blocks, False)
    torch.save(blocks.state_dict(), ckpt_path)
    builder = ModelBuilder(router=None,
                           tp_degree=8,
                           checkpoint_loader=get_ckpt,
                           compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir_blocks'))

    hidden_states = torch.rand([1, height * width // 256, 3072],
                               dtype=torch.bfloat16)
    encoder_hidden_states = torch.rand([1, max_sequence_length, 3072],
                                       dtype=torch.bfloat16)
    temb = torch.rand([1, 3072], dtype=torch.bfloat16)
    image_rotary_emb = torch.rand(
        [1, 1, height * width // 256 + max_sequence_length, 64, 2, 2],
        dtype=torch.bfloat16)
    sample_inputs = hidden_states, encoder_hidden_states, \
        temb, image_rotary_emb

    builder.add(key="blocks",
                model_instance=BaseModelInstance(partial(model.TracingTransformerBlockWrapper, transformer=transformer, transformerblock=transformer.transformer_blocks, is_distributed=True), input_output_aliases={}),
                example_inputs=[sample_inputs],
                compiler_args="""--target=trn2 --model-type=transformer --lnc=2 -O1 --tensorizer-options='--run-pg-layout-and-tiling'""")

    traced_model = builder.trace(initialize_model_weights=False)
    compiled_model_path = os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiled_model')
    if not os.path.exists(compiled_model_path):
        os.mkdir(compiled_model_path)
    torch.jit.save(traced_model, os.path.join(compiled_model_path, 'transformer_blocks.pt'))
    weights_path = os.path.join(COMPILER_WORKDIR_ROOT, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    builder.shard_checkpoint(os.path.join(weights_path, 'transformer_blocks'))


def generate_transformer_single_blocks_model(transformer, height, width, max_sequence_length):
    single_blocks = model.TracingSingleTransformerBlockWrapper(transformer, transformer.single_transformer_blocks, False)
    torch.save(single_blocks.state_dict(), ckpt_path)
    builder = ModelBuilder(router=None,
                           tp_degree=8,
                           checkpoint_loader=get_ckpt,
                           compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir_single_blocks'))

    hidden_states = torch.rand(
        [1, height * width // 256 + max_sequence_length, 3072],
        dtype=torch.bfloat16)
    temb = torch.rand([1, 3072], dtype=torch.bfloat16)
    image_rotary_emb = torch.rand(
        [1, 1, height * width // 256 + max_sequence_length, 64, 2, 2],
        dtype=torch.bfloat16)
    sample_inputs = hidden_states, temb, image_rotary_emb

    builder.add(key="single_blocks",
                model_instance=BaseModelInstance(partial(model.TracingSingleTransformerBlockWrapper, transformer=transformer, transformerblock=transformer.single_transformer_blocks, is_distributed=True), input_output_aliases={}),
                example_inputs=[sample_inputs],
                compiler_args="""--target=trn2 --model-type=transformer --lnc=2 -O1 --tensorizer-options='--run-pg-layout-and-tiling'""")

    traced_model = builder.trace(initialize_model_weights=False)
    compiled_model_path = os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiled_model')
    if not os.path.exists(compiled_model_path):
        os.mkdir(compiled_model_path)
    torch.jit.save(traced_model, os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiled_model/single_transformer_blocks.pt'))
    weights_path = os.path.join(COMPILER_WORKDIR_ROOT, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    builder.shard_checkpoint(os.path.join(weights_path, 'single_transformer_blocks'))


def generate_transformer_out_layers_model(transformer, height, width, max_sequence_length):
    out_layers = model.TracingTransformerOutLayerWrapper(transformer.norm_out, transformer.proj_out)
    torch.save(out_layers.state_dict(), ckpt_path)
    builder = ModelBuilder(router=None,
                           tp_degree=8,
                           checkpoint_loader=get_ckpt,
                           compiler_workdir=os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiler_workdir_out_layers'))

    hidden_states = torch.rand(
        [1, height * width // 256 + max_sequence_length, 3072],
        dtype=torch.bfloat16)
    encoder_hidden_states = torch.rand([1, max_sequence_length, 3072],
                                       dtype=torch.bfloat16)
    temb = torch.rand([1, 3072], dtype=torch.bfloat16)
    sample_inputs = hidden_states, encoder_hidden_states, temb

    builder.add(key="out_layers",
                model_instance=BaseModelInstance(partial(model.TracingTransformerOutLayerWrapper, norm_out=transformer.norm_out, proj_out=transformer.proj_out), input_output_aliases={}),
                example_inputs=[sample_inputs],
                compiler_args="""--target=trn2 --model-type=transformer --lnc=2 -O1 --tensorizer-options='--run-pg-layout-and-tiling'""")

    traced_model = builder.trace(initialize_model_weights=False)
    torch.jit.save(traced_model, os.path.join(COMPILER_WORKDIR_ROOT,
                                      'compiled_model/out_layers.pt'))
    weights_path = os.path.join(COMPILER_WORKDIR_ROOT, 'weights')
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    builder.shard_checkpoint(os.path.join(weights_path, 'out_layers'))


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
    parser.add_argument(
        "-m",
        "--max_sequence_length",
        type=int,
        default=512,
        help="maximum sequence length for the text embeddings"
    )
    args = parser.parse_args()
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev",
        torch_dtype=torch.bfloat16)
    transformer = copy.deepcopy(pipe.transformer)
    del pipe
    generate_transformer_model(transformer, args.height, args.width, args.max_sequence_length)
