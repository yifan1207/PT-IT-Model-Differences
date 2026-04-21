"""
Queue eval to run after training completes.

Polls the training volume every 5 min until all 12 probes are done (step 250),
then launches the eval pipeline.

    python scripts/modal_queue_eval_after_train.py
"""
import subprocess
import time
import json
import tempfile
import os

MODELS = ["gemma3_4b", "llama31_8b", "qwen3_4b", "mistral_7b", "deepseek_v2_lite", "olmo2_7b"]
VARIANTS = ["pt", "it"]
VOLUME = "0g-probes-v3"


def check_all_complete():
    """Check if all 12 training jobs are complete (step >= 250)."""
    complete = 0
    total = len(MODELS) * len(VARIANTS)

    for model in MODELS:
        for variant in VARIANTS:
            remote = f"{model}/{variant}/checkpoint.json"
            local = tempfile.mktemp(suffix=".json")
            try:
                result = subprocess.run(
                    ["modal", "volume", "get", VOLUME, remote, local, "--force"],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0 and os.path.exists(local):
                    with open(local) as f:
                        ckpt = json.load(f)
                    step = ckpt.get("checkpoint_step", 0)
                    if step >= 250:
                        complete += 1
                    else:
                        print(f"  {model}/{variant}: step {step}/250")
                else:
                    print(f"  {model}/{variant}: no checkpoint")
            except Exception as e:
                print(f"  {model}/{variant}: error {e}")
            finally:
                if os.path.exists(local):
                    os.unlink(local)

    return complete, total


def main():
    print("Waiting for all 12 training jobs to complete...")
    print(f"Polling volume '{VOLUME}' every 5 minutes.\n")

    while True:
        complete, total = check_all_complete()
        print(f"\n[{time.strftime('%H:%M')}] {complete}/{total} complete")

        if complete >= total:
            print("\n=== ALL TRAINING COMPLETE ===")
            print("Launching eval...")
            result = subprocess.run(
                ["modal", "run", "--detach", "scripts/modal_tuned_lens_eval.py"],
                capture_output=False,
            )
            print(f"Eval launched (exit code {result.returncode})")
            break

        time.sleep(300)  # 5 min


if __name__ == "__main__":
    main()
