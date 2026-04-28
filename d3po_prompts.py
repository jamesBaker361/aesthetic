from importlib import resources
import os
import functools
import random
from datasets import load_dataset

@functools.cache
def _load_lines(path)->list[str]:
    """
    Load lines from a file. First tries to load from `path` directly, and if that doesn't exist, searches the
    `d3po_pytorch/assets` directory for a file named `path`.
    """
    if not os.path.exists(path):
        path =os.path.join("d3po","d3po_pytorch","assets",path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Could not find {path} or ddpo_pytorch.assets/{path}")
    with open(path, "r") as f:
        return [line.strip() for line in f.readlines()]


def from_file(path, low=None, high=None):
    prompts = _load_lines(path)[low:high]
    return random.choice(prompts), {}


def imagenet_all():
    return from_file("imagenet_classes.txt")


def imagenet_animals():
    return from_file("imagenet_classes.txt", 0, 398)


def imagenet_dogs():
    return from_file("imagenet_classes.txt", 151, 269)


def simple_animals():
    return from_file("simple_animals.txt")

def anything_prompt():
    return from_file("anything_prompt.txt")

def unsafe_prompt():
    return from_file("unsafe_prompt.txt")

@functools.cache
def _load_nsfw_vocab():
    return [row["prompt"] for row in load_dataset("AIML-TUDA/i2p", split="train")]

def nsfw_prompts():
    return random.choice(_load_nsfw_vocab()), {}

@functools.cache
def _load_aesthetic_vocab():
    return [row["prompt"] for row in load_dataset("moonworks/lunara-aesthetic", split="train")]

def aesthetic_prompts():
    return random.choice(_load_aesthetic_vocab()), {}

def merged_prompts():
    return random.choice(_load_aesthetic_vocab() + _load_nsfw_vocab()), {}