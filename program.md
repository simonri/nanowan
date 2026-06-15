# nanowan

This is an experiment to have the LLM optimize video generation inference speed.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `jun11`). The branch `nanowan/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b nanowan/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `prepare.py` — fixed constants, correctness check, and summary output. Do not modify.
   - `run.py` — main inference script; the primary file to modify.
   - `model.py` — WAN transformer architecture.
   - `layers.py` — attention and other layer primitives.
   - `lora.py` — LoRA loading and application.
   - `scheduler.py` — flow matching Euler scheduler.
   - `vae.py` — VAE encoder/decoder.
   - `t5.py` — T5 text encoder.
   - `utils.py` — device, dtype, and memory utilities.
4. **Verify model weights exist**: Check that `models/` contains the diffusion models, VAE, text encoders, and LoRAs. If missing, tell the human to download them.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

You launch a run simply as: `uv run run.py`.

**What you CAN do:**
- Modify any `.py` file in the repo. Everything is fair game: attention computation, quantization, `torch.compile`, LoRA fusion, model precision, scheduler, number of denoising steps, loading strategy, kernel choices, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It contains the fixed constants, the correctness check (`compare_latents_ref`), and the summary output format (`print_summary`) — these are the ground truth.
- Modify model weight files in `models/`. They are read-only checkpoints.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.

**The goal is simple: get the lowest denoising_seconds.** The denoising loop is the compute bottleneck and the main lever for optimization. Everything is fair game. The only constraint is that the code produces a valid video without crashing.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful speedups, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small speedup that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome. When evaluating whether to keep a change, weigh the complexity cost against the speedup magnitude.

**The first run**: Your very first run should always be to establish the baseline, so you will run the script as is.

## Thinking before each experiment

Before touching any code, reason through the idea explicitly. This is not optional — skipping this step leads to wasted runs and shallow exploration.

**Write out:**

1. **Hypothesis**: What specifically are you changing, and what is the mechanism by which it should reduce `denoising_seconds`? Be precise — "this should be faster" is not a hypothesis. "Fusing the scale-shift into the norm kernel eliminates a separate elementwise pass over the activation tensor, saving one full read/write of ~X MB" is.

2. **Expected magnitude**: How much faster do you expect this to be, and why? Use the current baseline and known GPU throughput numbers (memory bandwidth, FLOP rate) to make a rough estimate. If you can't estimate even the sign, that's a signal the idea is undercooked.

3. **Risk assessment**: What could go wrong? Does this touch RMSE-sensitive paths? Could it OOM? Is it architecture-specific in a way that might not apply here?

4. **What success looks like**: Define in advance what result would cause you to keep vs discard this change. Don't decide after seeing the number.

## Prioritizing what to try

Don't randomly pick ideas. Reason about where the time is actually going before proposing experiments.

- **Profile first if you're unsure**: If you don't know where the bottleneck is, use `torch.profiler` or timing probes inside the denoising loop to find out before optimizing blindly.
- **Attack the biggest cost first**: A 10% improvement on a step that takes 80% of the time beats a 50% improvement on a step that takes 5%.
- **Build on what worked**: After a successful experiment, ask why it worked and whether there are related changes that exploit the same mechanism. Chain discoveries into a research thread rather than jumping randomly.
- **Learn from failures**: When an experiment doesn't help, update your mental model. What does that tell you about the bottleneck? Rule out classes of ideas, not just individual ones.
- **Distinguish memory-bound from compute-bound**: Many ops in the denoising loop are memory-bandwidth-limited, not FLOP-limited. Reducing memory traffic (fusion, lower precision, fewer passes) helps more than reducing FLOPs in those cases.

## Output format

Once the script finishes it prints a summary like this:

```
---
denoising_seconds:  38.90
total_seconds:     107.26
load_seconds:       65.24
decode_seconds:      3.12
peak_vram_mb:     45060.2
```

You can extract the key metric from the log file:

```
grep "^denoising_seconds:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	denoising_s	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. denoising_seconds (e.g. 38.90) — use 0.00 for crashes
3. peak memory in GB, round to .1f (e.g. 44.1 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	denoising_s	memory_gb	status	description
a1b2c3d	38.90	44.0	keep	baseline
b2c3d4e	32.00	44.2	keep	torch.compile denoising loop
c3d4e5f	40.00	44.0	discard	fp32 denoise (slower)
d4e5f6g	0.00	0.0	crash	INT8 quantization (kernel error)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `nanowan/jun11`).

LOOP FOREVER:

1. **Survey the state**: Check the current branch/commit and review `results.tsv` to understand what has been tried and what the current best is.
2. **Think before acting** (see "Thinking before each experiment" above): Write out your hypothesis, expected magnitude, risks, and success criteria before writing any code.
3. **Implement**: Tune a `.py` file with the chosen idea.
4. **Commit**: `git commit`
5. **Run**: `uv run run.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
6. **Read results**: `grep "^denoising_seconds:\|^peak_vram_mb:\|^latent_rmse:" run.log`
7. **Handle crashes**: If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up on this idea.
8. **Analyze**: Compare results to your pre-experiment prediction. Did it work the way you expected? If not, why? Update your mental model accordingly — understanding why something didn't work is as valuable as the result itself.
9. **Record**: Log results in `results.tsv` (do not commit this file — leave it untracked).
10. **Decide**:
    - If `denoising_seconds` improved → keep the commit, advance the branch.
    - If equal or worse → `git reset --hard HEAD~1` to discard.
11. **Plan the next experiment based on what you just learned**, not at random.

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard.

**Timeout**: A normal run takes ~100 seconds. If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~2 minutes (1 minute run + overhead) you can run ~30/hour, for a total of about 250 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
