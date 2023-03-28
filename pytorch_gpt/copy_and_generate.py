import argparse
import logging

import torch

from model import GPT
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import copy_model, generate_token

from hyperbox.mutables.spaces import ValueSpace
from hyperbox.networks.gpt import GPT2


logging.basicConfig(format="[%(levelname)s] %(asctime)s %(message)s")
logger = logging.getLogger(__file__)


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        "Copy weights of a HF model and generate text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "model_name",
        type=str,
        choices=("gpt2", "gpt2-medium", "gpt2-large", "distilgpt2"),
        help="Pretrained model to use",
    )
    parser.add_argument(
        "initial_text",
        type=str,
        help="Initial text",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="If True sample randomly otherwise take the most probable token",
    )
    parser.add_argument(
        "-s",
        "--steps",
        default=30,
        type=int,
        help="Number of new tokens to generate",
    )
    parser.add_argument("-r", "--random-state", type=int, help="Random state")
    parser.add_argument(
        "-t",
        "--temperature",
        default=1,
        type=float,
        help="Softmax logits temperature",
    )
    parser.add_argument(
        "-k",
        "--top-k",
        default=5,
        type=int,
        help="If specified, then selecting k most probable tokens",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="If True, then verbose"
    )

    args = parser.parse_args(argv)
    return args

def main(argv=None):
    """Copy weights and generate some text."""
    args = get_args(argv)

    # Setup logging
    if args.verbose:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)

    logger.info(f"CLI parameters: {vars(args)})")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model_official = AutoModelForCausalLM.from_pretrained(args.model_name)
    config_official = model_official.config

    our_params = [
        "vocab_size",
        "n_layer",
        "n_embd",
        "n_head",
        "n_positions",
        "attn_pdrop",
        "embd_pdrop",
        "resid_pdrop",
        "layer_norm_epsilon",
    ]

    config_ours = {k: getattr(config_official, k) for k in our_params}
    n_embed = config_ours["n_embd"]
    n_head = config_ours["n_head"]
    config_ours["n_embd"] = [n_embed//4, n_embed//2, n_embed]
    config_ours["n_head"] = [n_head//4, n_head//2, n_head]
    config_ours["n_embed"] = config_ours.pop("n_embd")
    logger.info(f"Model hyperparameters: {config_ours}")

    model_ours = GPT2(**config_ours) # from hyperbox
    model_ours = GPT(**config_ours) # from LLM-learn/pytorch_gpt
    model_ours.eval()

    print(sum(p.numel() for p in model_ours.parameters()))
    print(sum(p.numel() for p in model_official.parameters()))

    copy_model(model_official, model_ours)
    for name, param in model_ours.named_parameters():
        logger.info(f"{name}")
        if 'TM' in name:
            pass

    token_ixs = tokenizer(args.initial_text)["input_ids"]
    length = len(token_ixs)

    if args.random_state:
        torch.manual_seed(args.random_state)

    # Sample
    for step in range(args.steps):
        new_token_ix = generate_token(
            model_ours,
            token_ixs,
            sample=args.sample,
            top_k=args.top_k,
            temperature=args.temperature,
        )
        token_ixs.append(new_token_ix)

    text = tokenizer.decode(token_ixs)
    logger.info(f"Answer:\n{text}")


if __name__ == "__main__":
    main()
