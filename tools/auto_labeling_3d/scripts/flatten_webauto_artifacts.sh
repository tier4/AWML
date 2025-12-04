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

echo "Flattening WebAuto artifacts under: $ROOT_DIR"
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

    # Detect structure dir: prefer 0/, but also handle already-renamed WEBAUTO_STRUCTURE_0
    structure_dir=""
    if [ -d "$root_dir/0" ]; then
        structure_dir="$root_dir/0"
    elif [ -d "$root_dir/WEBAUTO_STRUCTURE_0" ]; then
        structure_dir="$root_dir/WEBAUTO_STRUCTURE_0"
    else
        echo "  -> No '0' or 'WEBAUTO_STRUCTURE_0' directory, skipping."
        echo
        continue
    fi

    # Base path where webauto puts intermediate artifacts
    base="$structure_dir/intermediate_artifacts/non_annotated_dataset"
    if [ ! -d "$base" ]; then
        echo "  -> '$base' not found, skipping."
        echo
        continue
    fi

    # Find the inner artifact directory (e.g. DB_J6Gen2_...)
    shopt -s nullglob
    candidates=( "$base"/* )
    shopt -u nullglob

    if [ "${#candidates[@]}" -eq 0 ]; then
        echo "  -> No artifact directory under '$base', skipping."
        echo
        continue
    fi

    # Use the first candidate (assuming only one artifact dir)
    artifact_dir="${candidates[0]}"
    echo "  -> Using artifact dir: $artifact_dir"

    # Move annotation directory if present and not already at root_dir
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

    # Move data directory if present and not already at root_dir
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

    # Move status.json if present and not already at root
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
        # If we got here via WEBAUTO_STRUCTURE_0, nothing to rename
        echo "  -> Structure dir is already 'WEBAUTO_STRUCTURE_0', leaving as-is."
    fi

    echo "  -> Done with $name"
    echo
done

echo "All done."
