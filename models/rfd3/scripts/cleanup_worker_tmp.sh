#!/usr/bin/env bash
# Free the multiprocessing temp dirs that DataLoader workers leak into TMPDIR.
#
# Why they leak
# -------------
# On Linux torch's default CPU-tensor sharing strategy is `file_descriptor`:
# every tensor that crosses a process boundary is passed by duplicating its file
# descriptor through `multiprocessing.resource_sharer`, which creates a unix
# socket inside `TMPDIR/pymp-XXXX/` (see
# torch/../multiprocessing/reductions.py:reduce_storage -> DupFd ->
# Listener(tempfile.mktemp(dir=util.get_temp_dir()))).
# `get_temp_dir()` calls `tempfile.mkdtemp(prefix='pymp-')` once per process and
# relies on an atexit hook to remove it. DataLoader workers are recycled every
# epoch and are terminated with a signal, so that hook never runs: each worker
# leaves a `pymp-*` directory behind. A 700-epoch sweep with 8 workers therefore
# accumulates ~5600 directories, and on a tmpfs each directory costs at least
# one page -> `OSError: [Errno 28] No space left on device: '/tmp/pymp-...'`
# in the queue feeder thread, which silently stops delivering batches.
#
# What this script does
# ---------------------
# Keeps the newest N `pymp-*` directories (those belong to the workers of the
# epoch currently running) and deletes the rest. Reports first, deletes only
# with --apply.
#
# Usage
# -----
#   bash models/rfd3/scripts/cleanup_worker_tmp.sh                     # report only
#   bash models/rfd3/scripts/cleanup_worker_tmp.sh --apply             # keep newest 8
#   bash models/rfd3/scripts/cleanup_worker_tmp.sh --keep 16 --apply
#   bash models/rfd3/scripts/cleanup_worker_tmp.sh --dir /tmp --top 15
#
# Notes
# -----
# * Deleting a directory that a LIVE worker is still using breaks that worker's
#   next descriptor transfer (its socket path disappears). That is why the
#   newest --keep dirs are preserved; they are the workers of the running epoch.
#   Set --keep to your number of dataloader workers (NUM_WORKERS, 8 for
#   PRESET=a6000) or slightly more.
# * A cleaner long-term fix is to point TMPDIR at a roomy filesystem before
#   starting the run (run_nmf_zkp_pdb_sweep.sh now does this), or to enable
#   persistent_workers so workers are not recycled every epoch.

set -uo pipefail

TARGET_DIR="/tmp"
KEEP=8
TOP=10
APPLY=0
MIN_AGE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dir)     TARGET_DIR="$2"; shift 2 ;;
    --dir=*)   TARGET_DIR="${1#*=}"; shift ;;
    --keep)    KEEP="$2"; shift 2 ;;
    --keep=*)  KEEP="${1#*=}"; shift ;;
    --top)     TOP="$2"; shift 2 ;;
    --top=*)   TOP="${1#*=}"; shift ;;
    --min-age) MIN_AGE="$2"; shift 2 ;;
    --apply)   APPLY=1; shift ;;
    -h|--help) sed -n '2,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

# Safety guard: only ever touch worker temp dirs below a temp-like directory.
case "${TARGET_DIR}" in
  /|/root|/home|"${HOME:-/nonexistent}"|/usr|/etc|/var|/backup01|/media|/mnt|/dev)
    echo "REFUSING to operate on ${TARGET_DIR}: pass a temp directory such as /tmp or <LOG_ROOT>/_worker_tmp." >&2
    exit 1 ;;
esac
if [[ ! -d "${TARGET_DIR}" ]]; then
  echo "ERROR: ${TARGET_DIR} is not a directory" >&2
  exit 1
fi

label() { printf '%s\n' "----------------------------------------------------------------"; }

label
echo "Filesystem for ${TARGET_DIR}"
df -h "${TARGET_DIR}" | sed 's/^/  /'
df -i "${TARGET_DIR}" | sed 's/^/  /'
label

# --------------------------------------------------------------------------- #
# Inventories (read-only)
# --------------------------------------------------------------------------- #
mapfile -t PYMP_DIRS < <(ls -dt "${TARGET_DIR}"/pymp-* 2>/dev/null)
total_pymp=${#PYMP_DIRS[@]}

echo "pymp-* directories: ${total_pymp}"
if (( total_pymp > 0 )); then
  if command -v du >/dev/null 2>&1; then
    echo -n "  bytes held by them: "
    du -sch "${PYMP_DIRS[@]}" 2>/dev/null | tail -1 | awk '{print $1}'
  fi
  echo "  newest: $(basename "${PYMP_DIRS[0]}")  ($(stat -c '%y' "${PYMP_DIRS[0]}" | cut -c1-19))"
  echo "  oldest: $(basename "${PYMP_DIRS[-1]}")  ($(stat -c '%y' "${PYMP_DIRS[-1]}" | cut -c1-19))"
  echo -n "  created in the last 60 min: "
  find "${TARGET_DIR}" -maxdepth 1 -name 'pymp-*' -mmin -60 2>/dev/null | wc -l
fi

echo
echo "Also worth a look (reported, never touched):"
if command -v du >/dev/null 2>&1; then
  du -sh "${TARGET_DIR}"/* 2>/dev/null | sort -rh | head -"${TOP}" | sed 's/^/  /'
fi
echo "  /dev/shm leftovers: $(ls -d /dev/shm/torch_* 2>/dev/null | wc -l) (file_system sharing strategy pattern)"

# --------------------------------------------------------------------------- #
# Candidate list
# --------------------------------------------------------------------------- #
CANDIDATES=()
if (( total_pymp > KEEP )); then
  CANDIDATES=("${PYMP_DIRS[@]:KEEP}")
fi

label
if (( ${#CANDIDATES[@]} == 0 )); then
  echo "Nothing to delete: ${total_pymp} pymp dir(s) <= --keep ${KEEP}."
  exit 0
fi

echo "Would delete ${#CANDIDATES[@]} of ${total_pymp} pymp dir(s) (keeping the newest ${KEEP})."
if (( MIN_AGE > 0 )); then
  echo "  (only those older than ${MIN_AGE} minute(s) -- the rest are kept regardless of --keep)"
fi
echo
echo "Sample targets (first 10):"
printf '  %s\n' "${CANDIDATES[@]:0:10}"
(( ${#CANDIDATES[@]} > 10 )) && echo "  ... and $(( ${#CANDIDATES[@]} - 10 )) more"

if [[ "${APPLY}" != "1" ]]; then
  echo
  echo "DRY RUN. Nothing was deleted."
  echo
  echo ">>> If the sweep process is still running, re-run with --keep <NUM_WORKERS> (>= 8) and --apply:"
  echo ">>>   bash models/rfd3/scripts/cleanup_worker_tmp.sh --keep ${KEEP} --apply"
  echo ">>> If nothing is training right now and you just want the space back:"
  echo ">>>   bash models/rfd3/scripts/cleanup_worker_tmp.sh --keep 0 --apply"
  exit 0
fi

label
echo "APPLYING: deleting ${#CANDIDATES[@]} pymp dir(s) from ${TARGET_DIR}"
before=$(df -h "${TARGET_DIR}" | awk 'NR==2 {print $4}')
deleted=0
failed=0
first_error=""
for path in "${CANDIDATES[@]}"; do
  if [[ -z "${MIN_AGE}" || "${MIN_AGE}" == "0" ]] || \
     [[ $(find "${path}" -maxdepth 0 -mmin +"${MIN_AGE}" 2>/dev/null | wc -l) -gt 0 ]]; then
    if rm -rf -- "${path}" 2>/tmp/.cleanup_err_$$; then
      deleted=$((deleted + 1))
    else
      failed=$((failed + 1))
      [[ -z "${first_error}" ]] && first_error="$(cat /tmp/.cleanup_err_$$ 2>/dev/null)"
    fi
  fi
done
if (( failed > 0 )); then
  echo "${failed} director(ies) could not be removed." >&2
  [[ -n "${first_error}" ]] && echo "  first error: ${first_error}" >&2
  echo "  (if it says 'Operation not permitted' or 'Read-only file system' the shell" >&2
  echo "   is sandboxed; run this script directly on the training host instead.)" >&2
fi
after=$(df -h "${TARGET_DIR}" | awk 'NR==2 {print $4}')
echo "Deleted ${deleted} dir(s)${failed:+, ${failed} failed}."
echo "Free space before: ${before}   after: ${after}"
df -h "${TARGET_DIR}" | sed 's/^/  /'
echo
echo "NOTE: this leak repeats every epoch (workers are recycled). If the sweep must"
echo "      keep running, either re-run this script periodically or restart the run"
echo "      with TMPDIR pointing at a roomy filesystem (run_nmf_zkp_pdb_sweep.sh"
echo "      sets WORKER_TMP for this)."
