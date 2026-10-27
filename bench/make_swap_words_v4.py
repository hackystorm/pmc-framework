# SPDX-License-Identifier: GPL-3.0-only
"""Build the frozen v4 word list for the next Substrate Swap attempt.

The v1, v2 and v3 lists are frozen and their evaluation facts must not be
used to design a later attempt.  This script therefore starts from a plain
pool of common words, removes every word that appears in v1, v2 or v3, keeps
only the words that both substrates tokenize as a single token, and lays the
survivors out with a fixed seed.

The output is the pre-registration: ``bench/swap_words_v4.json`` is written
once, its SHA-256 is recorded in the lab notes, and the attempt runs against
that file.  Re-running this script reproduces the same file byte for byte.

The script needs a patched ``libllama`` and the two model files; it is a
preparation tool, not part of the benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from pmc.runtime.live_spike import LlamaSpike

SPEC_PATH = Path(__file__).with_name("swap_words_v4.json")
SEED = 20261031
TRAINING_COUNT = 30
EVALUATION_COUNT = 8
CONTROL_COUNT = 8
WORD_COUNT = 84

POOL = (
    # plants and nature
    "acorn alder almond amber aspen bamboo bay beech berry birch blossom "
    "bramble briar brook cactus cedar clover coral cotton cypress dahlia dune "
    "elm fern fjord flax garlic ginger ginkgo grove hazel heather holly ivy "
    "juniper kelp lichen lilac lotus magnolia maple nectar nutmeg olive "
    "orchid palm peach pebble pine pollen poppy reed rosemary seaweed seed "
    "spruce stem thistle thorn thyme tulip vine walnut wheat willow yarrow "
    "zinnia moss"
    # animals
    "badger bat bear bee beetle bison camel cat crab crow deer dog dolphin "
    "donkey eagle elk falcon ferret fox frog goat hare hawk heron ibex "
    "iguana jaguar kestrel kitten koala lamb lizard lynx mink moose mouse "
    "newt otter owl panda pigeon pony puma rabbit raven rhino robin salmon "
    "seal shark sheep shrew skunk sloth sparrow spider swan tiger toad "
    "turtle viper walrus weasel whale wolf wombat worm wren yak zebra"
    # objects
    "anchor anvil apron arrow axe badge balloon banjo barrel basin basket "
    "baton beacon bell belt bench bicycle blanket blender board boat bolt "
    "book bottle bowl bracelet brick bridge brush bucket buckle bulb button "
    "cabin cable camera candle canoe cape card carpet cart castle chain chair "
    "chalk chest chisel clock cloth coin comb compass cord crate crayon crown "
    "cup curtain cushion dagger dial diary dish drum easel envelope fan "
    "faucet fence file flag flask flute folder fork frame funnel furnace "
    "gadget gate gauge gear glass globe glove glue hammer handle harp hat "
    "hinge hook horn hose jar jug kettle key kite knife ladder lamp lantern "
    "ledger lens lever magnet mallet map marble mask mast mat medal mirror "
    "mitten mug nail napkin needle net notebook oar pad paddle paint pan "
    "paper parcel pen pencil phone piano pillow pipe piston plate pliers "
    "pocket pot pouch printer pulley puzzle quilt racket radio raft ramp "
    "razor ribbon ring robe rocket rope ruler saddle sail sandal scarf screen "
    "screw shelf shield shirt shoe shovel sink skate sketch sled sleeve "
    "slide socket spool spoon stamp statue stool stove strap straw string "
    "suitcase sweater switch sword table tag tank tape teapot telescope tent "
    "thread ticket tile timer tire toaster tool torch towel tower tray tripod "
    "trophy truck trumpet tube tunnel umbrella valve vase vest wagon wallet "
    "wand watch wheel whistle window wire wrench yarn zipper"
    # food
    "biscuit butter candy cereal cheese cookie cream honey jam jelly ketchup "
    "lemon melon mustard noodle pancake pastry pickle pizza porridge pudding "
    "salad sauce soup syrup toast yogurt"
    # places
    "alley attic barn camp cellar chapel city coast desert field harbor hill "
    "jungle lagoon market meadow mill mine moor mountain museum orchard park "
    "pasture path peak pier plain pond port prairie quarry ranch reef ridge "
    "river road ruin shore slope spring summit swamp terrace thicket trail "
    "village volcano wetland yard"
).split()

TEMPLATES = {
    "fact": (
        "You are a terse assistant.\nUser: The code word for {cue} is {value}. "
        "Remember it.\nAssistant: Noted."
    ),
    "question": (
        "You are a terse assistant.\nUser: What is the code word for {cue}? "
        "Answer with one word.\nAssistant:"
    ),
    "reinjection": (
        "You are a terse assistant.\nUser: The code word for {cue} is {value}. "
        "Remember it.\nAssistant: Noted.\nUser: What is the code word for "
        "{cue}? Answer with one word.\nAssistant:"
    ),
}


def used_words(paths: Iterable[str | Path]) -> set[str]:
    """Every word that already appears in a frozen list."""
    words: set[str] = set()
    for path in paths:
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        words.update(spec["words"])
        for fact in spec["training_facts"] + spec["evaluation_facts"]:
            words.add(fact["cue"])
            words.add(fact["value"])
        for control in spec["control_questions"]:
            words.add(control["cue"])
    return words


def single_token_words(
    pool: Iterable[str], lib: str, shim: str, models: dict[str, str]
) -> list[str]:
    """Keep the words that are one token after a space on every host."""
    candidates = sorted(set(pool))
    keep: list[str] = []
    for host, model in models.items():
        spike = LlamaSpike(lib, shim, model)
        try:
            for word in candidates:
                tokens = spike.tokenize(" " + word)
                if len(tokens) != 2:
                    candidates.remove(word)
        finally:
            spike.close()
        print(f"{host}: {len(candidates)} single-token candidates left", flush=True)
    return candidates


def build_spec(words: list[str]) -> dict[str, Any]:
    """Lay the word list out into the frozen split."""
    if len(words) != WORD_COUNT:
        raise ValueError(f"the split needs {WORD_COUNT} words, got {len(words)}")
    cursor = 0

    def take(count: int) -> list[str]:
        nonlocal cursor
        chunk = words[cursor : cursor + count]
        cursor += count
        return chunk

    training_cues = take(TRAINING_COUNT)
    training_values = take(TRAINING_COUNT)
    evaluation_cues = take(EVALUATION_COUNT)
    evaluation_values = take(EVALUATION_COUNT)
    control_cues = take(CONTROL_COUNT)
    return {
        "schema": "pmc-swap-words/1",
        "mechanism": "pmc-swap-shared-body/1",
        "training_count": TRAINING_COUNT,
        "word_count": WORD_COUNT,
        "seed": SEED,
        "hosts": ["350m", "8b"],
        "anchor": "cue_delta",
        "neutral_cue": "thing",
        "words": list(words),
        "training_facts": [
            {"cue": cue, "value": value}
            for cue, value in zip(training_cues, training_values)
        ],
        "evaluation_facts": [
            {"cue": cue, "value": value}
            for cue, value in zip(evaluation_cues, evaluation_values)
        ],
        "control_questions": [
            {"cue": cue, "paired_eval": index}
            for index, cue in enumerate(control_cues)
        ],
        "templates": dict(TEMPLATES),
        "projection": {
            "input_dim": {"350m": 1024, "8b": 2048},
            "output_dim": 128,
            "seed": SEED,
        },
        "adapter": {
            "output_dim": 64,
            "temperature": 0.1,
            "epochs": 300,
            "early_stopping": False,
            "learning_rate": 0.01,
            "weight_decay": 0.0001,
            "normalize_input": True,
        },
        "bias": 200.0,
        "state_schema": "pmc.state.swap/1",
        "arms": ["pmc_a", "pmc_b"],
        "pass": {
            "host_a_min": 5,
            "host_b_min": 4,
            "host_b_margin": 1,
            "no_state_max": 1,
            "fresh_max": 2,
            "control_max": 2,
            "note": (
                "host B must be at least host A minus 1; every miss is reported"
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lib", required=True, help="patched libllama")
    parser.add_argument("--shim", required=True, help="pmc spike shim")
    parser.add_argument("--model-350m", required=True)
    parser.add_argument("--model-8b", required=True)
    parser.add_argument(
        "--frozen",
        nargs="*",
        default=[
            str(Path(__file__).with_name("swap_words.json")),
            str(Path(__file__).with_name("swap_words_v2.json")),
            str(Path(__file__).with_name("swap_words_v3.json")),
        ],
        help="frozen lists whose words must not be reused",
    )
    parser.add_argument("--out", default=str(SPEC_PATH))
    args = parser.parse_args(argv)

    excluded = used_words(args.frozen)
    pool = [word for word in POOL if word not in excluded]
    print(f"pool: {len(POOL)} words, {len(excluded)} already used", flush=True)
    candidates = single_token_words(
        pool, args.lib, args.shim, {"350m": args.model_350m, "8b": args.model_8b}
    )
    if len(candidates) < WORD_COUNT:
        raise SystemExit(
            f"only {len(candidates)} usable words, the split needs {WORD_COUNT}"
        )
    rng = random.Random(SEED)
    shuffled = candidates[:]
    rng.shuffle(shuffled)
    chosen = sorted(shuffled[:WORD_COUNT])
    spec = build_spec(chosen)
    payload = json.dumps(spec, indent=2, sort_keys=True) + "\n"
    Path(args.out).write_text(payload, encoding="utf-8")
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    print(f"wrote {args.out}")
    print(f"spec sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
