#!/bin/bash
# Sync chain data from cluster.
# Usage: grab_data.sh <project-dir>        (looks for sync.conf inside)
#        grab_data.sh <path/to/sync.conf>  (direct path)
#        grab_data.sh                      (looks for sync.conf in current directory)
#
# The conf file defines REMOTE, LOCAL, and DIRS.
# LOCAL is resolved relative to the directory containing the conf file,
# so you can call this script from anywhere.

set -e

ARG="${1:-.}"

# Accept either a directory or a direct path to the conf file
if [ -d "$ARG" ]; then
    CONF="$ARG/sync.conf"
else
    CONF="$ARG"
fi

if [ ! -f "$CONF" ]; then
    echo "Error: no sync.conf found at $CONF"
    echo "Usage: $0 <project-dir>  or  $0 <path/to/sync.conf>"
    exit 1
fi

source "$CONF"

# Validate required variables
for var in REMOTE LOCAL DIRS; do
    if [ -z "${!var+x}" ]; then
        echo "Error: '$var' not defined in $CONF"
        exit 1
    fi
done

# Resolve LOCAL relative to the conf file's directory
CONF_DIR="$(dirname "$(realpath "$CONF")")"
LOCAL_ABS="$CONF_DIR/$LOCAL"
mkdir -p "$LOCAL_ABS"

echo "Remote : $REMOTE"
echo "Local  : $LOCAL_ABS"
echo ""

for dir in "${DIRS[@]}"; do
    echo "==================================================="
    echo "Syncing $dir..."
    echo "==================================================="
    rsync -azuve ssh "$REMOTE/$dir" "$LOCAL_ABS/"
done

echo ""
echo "All syncs completed"
