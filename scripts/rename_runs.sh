#!/usr/bin/env bash
# Rename existing detection and SPNet run directories to put UTC time first.
set -euo pipefail
shopt -s nullglob

usage='Usage: rename_runs.sh [--dry-run] [runs_dir]'
if [[ ${1:-} == --help ]]; then
    printf '%s\n' "$usage"
    exit 0
fi

dry_run=false
if [[ ${1:-} == --dry-run ]]; then
    dry_run=true
    shift
fi
if (( $# > 1 )); then
    printf '%s\n' "$usage" >&2
    exit 2
fi

runs_dir=${1:-"$(dirname "${BASH_SOURCE[0]}")/../runs"}
if [[ ! -d $runs_dir ]]; then
    printf 'Run directory does not exist: %s\n' "$runs_dir" >&2
    exit 1
fi

legacy_name='^([0-9]{8})T([0-9]{6})Z_(detection_match[0-9]+|[[:alpha:]]+_[0-9]+x[0-9]+_match[0-9]+)_([0-9a-f]{8})$'
found=false
status=0
for source in "$runs_dir"/*; do
    name=${source##*/}
    if [[ ! -d $source || -L $source || ! $name =~ $legacy_name ]]; then
        continue
    fi
    found=true

    date=${BASH_REMATCH[1]}
    time=${BASH_REMATCH[2]}
    run_label=${BASH_REMATCH[3]}
    run_id=${BASH_REMATCH[4]}
    date="${date:0:4}-${date:4:2}-${date:6:2}"
    time="${time:0:2}-${time:2:2}-${time:4:2}"
    new_name="${time}_${run_label}_${date}_${run_id}"
    target="$runs_dir/$new_name"

    if [[ -e $target || -L $target ]]; then
        printf 'Skipped %s because %s already exists\n' "$name" "$new_name" >&2
        status=1
        continue
    fi

    if [[ $dry_run == false ]]; then
        if ! mv -nT -- "$source" "$target" || [[ -e $source ]]; then
            printf 'Could not rename %s to %s\n' "$name" "$new_name" >&2
            status=1
            continue
        fi
    fi
    printf '%s -> %s\n' "$name" "$new_name"
done

if [[ $found == false ]]; then
    printf 'No legacy run directories found.\n'
fi
exit "$status"
