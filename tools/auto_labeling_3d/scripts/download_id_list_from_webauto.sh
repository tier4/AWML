#!/usr/bin/env bash
set -euo pipefail

# Default options
FLATTEN=true

# --- Option parsing (only --flatten=...) ---

POSITIONAL=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --flatten=*)
            val="${1#*=}"
            case "$val" in
                true|TRUE|1|yes|y) FLATTEN=true ;;
                false|FALSE|0|no|n) FLATTEN=false ;;
                *)
                    echo "Invalid value for --flatten: $val (use true/false)" >&2
                    exit 1
                    ;;
            esac
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [--flatten=true|false] ID_FILE OUTPUT_DIR [PROJECT_ID] [MAX_JOBS]" >&2
            echo "  ID_FILE     : Text file with one annotation-dataset-id per line" >&2
            echo "  OUTPUT_DIR  : Destination directory for downloaded assets" >&2
            echo "  PROJECT_ID  : (optional) WebAuto project-id, default: x2_dev" >&2
            echo "  MAX_JOBS    : (optional) parallel downloads, default: 5" >&2
            exit 0
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL[@]}"

if [ "$#" -lt 2 ] || [ "$#" -gt 4 ]; then
    echo "Usage: $0 [--flatten=true|false] ID_FILE OUTPUT_DIR [PROJECT_ID := x2_dev] [MAX_JOBS := 5]" >&2
    exit 1
fi

ID_FILE="$1"
OUTPUT_DIR="$2"
PROJECT_ID="${3:-x2_dev}"
MAX_JOBS="${4:-5}"
TYPE="non_annotated_dataset"

if [ ! -f "$ID_FILE" ]; then
    echo "ID file not found: $ID_FILE" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Read IDs into an array (ignore empty lines and comments)
mapfile -t IDS < <(grep -Ev '^\s*($|#)' "$ID_FILE" || true)
TOTAL=${#IDS[@]}

if [ "$TOTAL" -eq 0 ]; then
    echo "No IDs found in $ID_FILE" >&2
    exit 1
fi

echo "ID file      : $ID_FILE"
echo "Output dir   : $OUTPUT_DIR"
echo "Project ID   : $PROJECT_ID"
echo "Type         : $TYPE"
echo "Parallel jobs: $MAX_JOBS"
echo "Total IDs    : $TOTAL"
echo "Flatten      : $FLATTEN"
echo

# Temp file to track completion status
DONE_FILE="$(mktemp)"
LOG_FILE="$(mktemp)"
trap 'rm -f "$DONE_FILE" "$LOG_FILE"' EXIT

download_one() {
    local id="$1"
    local status="OK"

    echo "[START] $id" >> "$LOG_FILE"

    if ! webauto data annotation-dataset pull-intermediate-artifact \
        --project-id "$PROJECT_ID" \
        --annotation-dataset-id "$id" \
        --type "$TYPE" \
        --asset-dir "$OUTPUT_DIR"; then
        status="FAIL"
    fi

    echo "$status $id" >> "$DONE_FILE"
    echo "[DONE]  $id -> $status" >> "$LOG_FILE"
}

# Progress bar printer (runs in main process)
print_progress() {
    local completed percent width filled empty bar
    width=40

    while :; do
        if [ -f "$DONE_FILE" ]; then
            completed=$(wc -l < "$DONE_FILE" 2>/dev/null || echo 0)
        else
            completed=0
        fi

        if [ "$TOTAL" -le 0 ]; then
            percent=0
        else
            percent=$(( completed * 100 / TOTAL ))
        fi

        filled=$(( percent * width / 100 ))
        empty=$(( width - filled ))

        bar="$(printf '#%.0s' $(seq 1 "$filled" 2>/dev/null || true))"
        bar="$bar$(printf ' %.0s' $(seq 1 "$empty" 2>/dev/null || true))"

        printf "\r[%s] %3d%% (%d/%d) completed" "$bar" "$percent" "$completed" "$TOTAL"

        if [ "$completed" -ge "$TOTAL" ]; then
            break
        fi

        sleep 1
    done

    echo
}

# Flatten a single dataset directory: OUTPUT_DIR/<id>
flatten_one() {
    local root_dir="$1"

    if [ ! -d "$root_dir" ]; then
        echo "  -> '$root_dir' not found, skipping."
        return
    fi

    # Detect structure dir: prefer 0/, but also handle already-renamed WEBAUTO_STRUCTURE_0
    local structure_dir=""
    if [ -d "$root_dir/0" ]; then
        structure_dir="$root_dir/0"
    elif [ -d "$root_dir/WEBAUTO_STRUCTURE_0" ]; then
        structure_dir="$root_dir/WEBAUTO_STRUCTURE_0"
    else
        echo "  -> No '0' or 'WEBAUTO_STRUCTURE_0' directory, skipping flatten."
        return
    fi

    local base="$structure_dir/intermediate_artifacts/non_annotated_dataset"
    if [ ! -d "$base" ]; then
        echo "  -> '$base' not found, skipping flatten."
        return
    fi

    shopt -s nullglob
    local candidates=( "$base"/* )
    shopt -u nullglob

    if [ "${#candidates[@]}" -eq 0 ]; then
        echo "  -> No artifact directory under '$base', skipping flatten."
        return
    fi

    local artifact_dir="${candidates[0]}"
    echo "  -> Using artifact dir: $artifact_dir"

    # Move annotation/ to root_dir if present
    if [ -d "$artifact_dir/annotation" ]; then
        if [ -e "$root_dir/annotation" ]; then
            echo "  -> 'annotation/' already exists at root, leaving as-is."
        else
            echo "  -> Moving annotation/ to $root_dir/"
            mv "$artifact_dir/annotation" "$root_dir/"
        fi
    else
        echo "  -> No annotation/ found in artifact dir."
    fi

    # Move data/ to root_dir if present
    if [ -d "$artifact_dir/data" ]; then
        if [ -e "$root_dir/data" ]; then
            echo "  -> 'data/' already exists at root, leaving as-is."
        else
            echo "  -> Moving data/ to $root_dir/"
            mv "$artifact_dir/data" "$root_dir/"
        fi
    else
        echo "  -> No data/ found in artifact dir."
    fi

    # Move status.json to root_dir if present
    if [ -f "$artifact_dir/status.json" ]; then
        if [ -e "$root_dir/status.json" ]; then
            echo "  -> 'status.json' already exists at root, leaving as-is."
        else
            echo "  -> Moving status.json to $root_dir/"
            mv "$artifact_dir/status.json" "$root_dir/"
        fi
    else
        echo "  -> No status.json found in artifact dir."
    fi

    # After flattening, rename 0/ → WEBAUTO_STRUCTURE_0 (if applicable)
    if [ -d "$root_dir/0" ]; then
        if [ -e "$root_dir/WEBAUTO_STRUCTURE_0" ]; then
            echo "  -> 'WEBAUTO_STRUCTURE_0' already exists, not renaming '0/'."
        else
            echo "  -> Renaming '0/' to 'WEBAUTO_STRUCTURE_0/'."
            mv "$root_dir/0" "$root_dir/WEBAUTO_STRUCTURE_0"
        fi
    else
        echo "  -> Structure dir is already 'WEBAUTO_STRUCTURE_0', leaving as-is."
    fi
}

# --- Start downloads ---

print_progress &
PROGRESS_PID=$!

for id in "${IDS[@]}"; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_JOBS" ]; do
        sleep 0.5
    done
    download_one "$id" &
done

wait
wait "$PROGRESS_PID" 2>/dev/null || true

echo
echo "=== Download results ==="
sort "$DONE_FILE"
echo

# --- Flatten phase (optional) ---

if [ "$FLATTEN" = true ]; then
    echo "Flattening downloaded datasets..."
    for id in "${IDS[@]}"; do
        echo "Processing $id ..."
        flatten_one "$OUTPUT_DIR/$id"
        echo
    done
    echo "Flattening done."
else
    echo "Flattening disabled (--flatten=false)."
fi

echo
echo "Detailed log at: $LOG_FILE"
