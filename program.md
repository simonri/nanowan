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

The TSV has a header row and 6 columns:

```
commit	denoising_s	memory_gb	latent_rmse	status	description
```

1. git commit hash (short, 7 chars)
2. denoising_seconds (e.g. 38.90) — use 0.00 for crashes
3. peak memory in GB, round to .1f (e.g. 44.1 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. latent_rmse from `grep "^latent_rmse:" run.log` — use N/A for crashes
5. status: `keep`, `discard`, or `crash`
6. short text description of what this experiment tried

Example:

```
commit	denoising_s	memory_gb	latent_rmse	status	description
a1b2c3d	38.90	44.0	0.000000	keep	baseline
b2c3d4e	32.00	44.2	0.001234	keep	torch.compile denoising loop
c3d4e5f	40.00	44.0	0.350000	discard	fp8 (RMSE too high)
d4e5f6g	0.00	0.0	N/A	crash	INT8 quantization (kernel error)
```

**Quality constraint**: `latent_rmse` must stay below **0.15**. Changes that push RMSE above 0.15 must be reverted even if they are faster. Log the RMSE for every experiment — a "keep" result must satisfy BOTH lower denoising_seconds AND latent_rmse < 0.15.

## The experiment loop

The experiment runs on a dedicated branch (e.g. `nanowan/jun11`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Tune a `.py` file with an experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run run.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^denoising_seconds:\|^peak_vram_mb:\|^latent_rmse:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. If denoising_seconds improved (lower), you "advance" the branch, keeping the git commit
9. If denoising_seconds is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard.

**Timeout**: A normal run takes ~100 seconds. If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~2 minutes (1 minute run + overhead) you can run ~30/hour, for a total of about 250 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
