# SPDX-License-Identifier: GPL-3.0-only

# Benchmarks

The evaluation battery. Its job is to fail as easily as it passes.

## What is measured

| Metric | Question it answers |
|---|---|
| recall at k | can it retrieve what it consolidated? |
| forgetting rate | what did consolidation cost? |
| false-memory rate | does it confabulate? |
| interference rate | what did the new material break? |
| expected calibration error | does its confidence match its accuracy? |
| Brier score | the same question, squared |
| rumination pressure | is it looping rather than remembering? |
| sidecar overhead | how large is the public record next to the private interior? |

Every one of them measures **function**. None of them measures
consciousness, and the harness says so in its own output.

## Running it

```bash
python3 bench/run_eval.py --gold gold.json --predictions predictions.json
python3 bench/run_eval.py --gold gold.json --predictions predictions.json \
    --baseline baseline.json --k 5
```

The formats are documented in the header of `run_eval.py`.

## Rules for a result

1. A result without a baseline is not a result.
2. A result without a seed and a configuration is not reproducible.
3. A result that only holds for the model that produced it is a result
   about the model, not about the method. Judge it independently.
4. Negative results get published with the same care as positive ones.

## What does not exist yet

There are no published numbers. No checkpoint has been trained, and the
Kaggle notebook does not exist. When the first real evaluation lands, it
lands here, with its configuration, its seed, its manifest and its
ablations.
