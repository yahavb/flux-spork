# FLUX.1 - dev
## Introduction
FLUX is one of the newest text-to-image generators available, and this variant "dev", out of "pro" [only available through API], "dev", and "schnell", offers cutting-edge output quality with competitive prompt following. In particular are details such as fingers or text. It is composed of the CLIP text encoder, the T5 text encoder, sent to a transformer, and finally goes through a variational autoencoder to decode the output.
## Compilation on Neuron
### Strategies
The current version depends primarily on tensor parallelism in the transformer. In particular, all linear layers EXCEPT the context embedder, cross embedder, and any layers in normalizations are parallelized by column, unless if it's in pairs, in which case the second layer of the pair is parallelized by row, without taking in all_gather from the first layer. A patch is used in the transformer for the design of the "single_transformer_blocks" in the transformer, fixing an accuracy issue happening with parallelism (column-parallelized results were not all-gathered before they were concatenated with other column-parallelized results, and sent into a row-parallelized layer; since all_gather is too expensive, row-parallelized layer was split for the respective input tensors and summed together, resulting in correct value).
### Compiler Flags in Use
The compilation bash script is currently configured for a 1024x1024 image, 512 sequence length (max possible), and 50 inference steps per image.
For the transformer specifically, `--model-type=unet-inference` is used.
For the text encoders, `--enable-fast-loading-neuron-binaries` is used.
For the VAE decoder, `--model-type=unet-inference` is used.
## How to Compile
Install `diffusers`, `transformers`, `sentencepiece`. Ensure you are on a TRN1 instance, on the `aws_neuron_venv_pytorch` environment (`source /home/ubuntu/aws_neuron_venv_pytorch/bin/activate`), and then run the `compile.sh` script found in `flux/trn1/` of this package. It is recommended to use `screen` to run this, as compilation is expected to take 1.5 hours.
## How to Run Basic Accuracy Checks
Only after the above compilation is complete then can the basic accuracy checks run. For this, run `test.sh` in the `flux/trn1/` of this package.
## How to Inference
The prompt can be modified in `inference.sh` as an argument sent to the `inference.py` file (the default prompt is "A cat holding a sign that says hello world"). Unless the dimensions or sequence length have been modified in the `compile.sh` script, do NOT change the respective values in the `inference.sh` script. However, the number of inference steps are allowed to be modified (default 50). Run the `inference.sh` script.# flux-spork
# flux-spork
# flux-spork
