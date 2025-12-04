#!/usr/bin/env bash
set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 ROOT_DIR"
    exit 1
fi

# Strip trailing slash if present
ROOT_DIR="${1%/}"

if [ ! -d "$ROOT_DIR" ]; then
    echo "Root directory '$ROOT_DIR' not found or not a directory."
    exit 1
fi

echo "Unflattening WebAuto artifacts under: $ROOT_DIR"
echo

# Iterate over all immediate subdirectories in ROOT_DIR
shopt -s nullglob
subdirs=( "$ROOT_DIR"/* )
shopt -u nullglob

if [ ${#subdirs[@]} -eq 0 ]; then
    echo "No subdirectories found under '$ROOT_DIR'. Nothing to do."
    exit 0
fi

for root_dir in "${subdirs[@]}"; do
    # Only process directories
    if [ ! -d "$root_dir" ]; then
        continue
    fi

    name="$(basename "$root_dir")"
    echo "Processing $name ..."

    ann_src="$root_dir/annotation"
    data_src="$root_dir/data"
    status_src="$root_dir/status.json"

    # If none of the flattened items exist, nothing to do for this dir
    if [ ! -d "$ann_src" ] && [ ! -d "$data_src" ] && [ ! -f "$status_src" ]; then
        echo "  -> No annotation/, data/, or status.json at root, skipping."
        echo
        continue
    fi

    # Detect structure dir: prefer WEBAUTO_STRUCTURE_0, but also support plain 0
    structure_dir=""
    used_placeholder=false
    if [ -d "$root_dir/WEBAUTO_STRUCTURE_0" ]; then
        structure_dir="$root_dir/WEBAUTO_STRUCTURE_0"
        used_placeholder=true
    elif [ -d "$root_dir/0" ]; then
        structure_dir="$root_dir/0"
        used_placeholder=false
    else
        echo "  -> No 'WEBAUTO_STRUCTURE_0' or '0' directory, cannot restore structure. Skipping."
        echo
        continue
    fi

    base="$structure_dir/intermediate_artifacts/non_annotated_dataset"
    if [ ! -d "$base" ]; then
        echo "  -> '$base' not found, cannot restore structure. Skipping."
        echo
        continue
    fi

    # Find the inner artifact directory (e.g. DB_J6Gen2_...)
    shopt -s nullglob
    candidates=( "$base"/* )
    shopt -u nullglob

    if [ "${#candidates[@]}" -eq 0 ]; then
        echo "  -> No artifact directory under '$base', cannot restore. Skipping."
        echo
        continue
    fi

    artifact_dir="${candidates[0]}"
    echo "  -> Using artifact dir: $artifact_dir"

    # Move annotation back if present at root and not already in artifact_dir
    if [ -d "$ann_src" ]; then
        if [ -e "$artifact_dir/annotation" ]; then
            echo "  -> 'annotation/' already exists in artifact dir, leaving both as-is."
        else
            echo "  -> Moving annotation/ back into artifact dir."
            mv "$ann_src" "$artifact_dir/"
        fi
    else
        echo "  -> No annotation/ at root."
    fi

    # Move data back if present at root and not already in artifact_dir
    if [ -d "$data_src" ]; then
        if [ -e "$artifact_dir/data" ]; then
            echo "  -> 'data/' already exists in artifact dir, leaving both as-is."
        else
            echo "  -> Moving data/ back into artifact dir."
            mv "$data_src" "$artifact_dir/"
        fi
    else
        echo "  -> No data/ at root."
    fi

    # Move status.json back if present at root and not already in artifact_dir
    if [ -f "$status_src" ]; then
        if [ -e "$artifact_dir/status.json" ]; then
            echo "  -> 'status.json' already exists in artifact dir, leaving both as-is."
        else
            echo "  -> Moving status.json back into artifact dir."
            mv "$status_src" "$artifact_dir/"
        fi
    else
        echo "  -> No status.json at root."
    fi

    # If we used the placeholder dir, try to rename it back to 0
    if $used_placeholder; then
        if [ -e "$root_dir/0" ]; then
            echo "  -> '0/' already exists, not renaming 'WEBAUTO_STRUCTURE_0/'."
        else
            echo "  -> Renaming 'WEBAUTO_STRUCTURE_0/' back to '0/'."
            mv "$root_dir/WEBAUTO_STRUCTURE_0" "$root_dir/0"
        fi
    fi

    echo "  -> Done with $name"
    echo
done

echo "All done."
