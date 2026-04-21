#!/usr/bin/env bash
# Upload completed experiment results to GCS and delete local copies.
#
# Prerequisites:
#   gcloud auth login           (interactive browser auth)
#   -- OR --
#   gcloud auth activate-service-account --key-file=/path/to/key.json
#
# Usage:
#   bash scripts/upload_to_gcs.sh [--dry-run]
#
# With --dry-run: prints what would be uploaded/deleted without doing anything.
# Without flags:  uploads then deletes local copies (irreversible — uploads verified first).

set -euo pipefail

BUCKET="gs://pt-vs-it-results"
LOCAL_RESULTS="$(cd "$(dirname "$0")/.." && pwd)/results"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=true
    echo "[dry-run] No files will be uploaded or deleted."
fi

# ── Sanity checks ─────────────────────────────────────────────────────────────
if ! command -v gsutil &>/dev/null && ! command -v gcloud &>/dev/null; then
    echo "ERROR: neither gsutil nor gcloud found."
    echo "Install the Google Cloud SDK:"
    echo "  curl https://sdk.cloud.google.com | bash"
    echo "  source ~/.bashrc"
    echo "  gcloud auth login"
    exit 1
fi

# Prefer gcloud storage (faster, parallel) over legacy gsutil if available.
if command -v gcloud &>/dev/null && gcloud storage --version &>/dev/null 2>&1; then
    UPLOAD_CMD="gcloud storage cp -r"
    LS_CMD="gcloud storage ls"
else
    UPLOAD_CMD="gsutil -m cp -r"
    LS_CMD="gsutil ls"
fi

echo "========================================"
echo " Upload results → GCS"
echo " bucket : ${BUCKET}"
echo " source : ${LOCAL_RESULTS}"
echo " tool   : ${UPLOAD_CMD%% *}"
echo " $(date)"
echo "========================================"

# ── What to upload ────────────────────────────────────────────────────────────
# Format: "local_path  gcs_destination  delete_after_upload"
# Plots are intentionally kept local (tiny, frequently referenced).
declare -a JOBS=(
    "${LOCAL_RESULTS}/exp2             ${BUCKET}/results/exp02_ic_ooc_reasoning_mechanistic_comparison              yes"
    "${LOCAL_RESULTS}/exp3             ${BUCKET}/results/exp03_corrective_stage_characterization              yes"
    "${LOCAL_RESULTS}/exp4             ${BUCKET}/results/exp04_phase_transition_characterization              yes"
    "${LOCAL_RESULTS}/poc_results.json ${BUCKET}/results/poc_results.json  yes"
    "${LOCAL_RESULTS}/weight_shift     ${BUCKET}/results/weight_shift      yes"
)

total_freed_gb=0

for job in "${JOBS[@]}"; do
    read -r local_path gcs_dest delete <<< "${job}"

    if [ ! -e "${local_path}" ]; then
        echo "[skip] ${local_path} — not found locally"
        continue
    fi

    size=$(du -sh "${local_path}" 2>/dev/null | cut -f1)
    echo ""
    echo "── ${local_path##*/} (${size}) ──────────────────────────"
    echo "   local  : ${local_path}"
    echo "   remote : ${gcs_dest}"

    if $DRY_RUN; then
        echo "   [dry-run] would upload then delete"
        continue
    fi

    # Upload
    echo "   uploading ..."
    ${UPLOAD_CMD} "${local_path}" "${gcs_dest}"

    # Verify the upload landed in the bucket
    echo "   verifying ..."
    if ! ${LS_CMD} "${gcs_dest}" &>/dev/null; then
        echo "   ERROR: verification failed — NOT deleting local copy"
        continue
    fi

    echo "   upload verified ✓"

    # Delete local copy
    if [[ "${delete}" == "yes" ]]; then
        rm -rf "${local_path}"
        echo "   deleted local copy"
        # rough GB freed
        num=$(echo "${size}" | grep -oP '[0-9.]+')
        unit=$(echo "${size}" | grep -oP '[A-Z]+')
        if [[ "${unit}" == "G" ]]; then
            total_freed_gb=$(echo "${total_freed_gb} + ${num}" | bc)
        fi
    fi
done

# ── Delete GPT-2 from HF cache (not used by exp5) ────────────────────────────
GPT2_CACHE="${HOME}/.cache/huggingface/hub/models--gpt2"
if [ -d "${GPT2_CACHE}" ]; then
    gpt2_size=$(du -sh "${GPT2_CACHE}" | cut -f1)
    echo ""
    echo "── GPT-2 HF cache (${gpt2_size}) ──────────────────────────"
    if $DRY_RUN; then
        echo "   [dry-run] would delete ${GPT2_CACHE}"
    else
        rm -rf "${GPT2_CACHE}"
        echo "   deleted ${GPT2_CACHE}"
    fi
fi

echo ""
echo "========================================"
if $DRY_RUN; then
    echo " Dry run complete — no changes made."
else
    echo " Done — ~${total_freed_gb} GB freed locally."
    df -h / | tail -1
fi
echo " Results browsable at:"
echo "   https://console.cloud.google.com/storage/browser/pt-vs-it-results"
echo "========================================"
