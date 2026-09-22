#!/bin/sh
# Eval CI: build a model from the repo's own source (self-contained,
# no external corpus needed), evaluate the committed holdout set, and
# fail if hit@1 regresses beyond the tolerance vs the saved baseline.
#
# Regenerate the baseline after intentional model/format changes:
#   scripts/evalci.sh --write-baseline
set -e
cd "$(dirname "$0")/.."

WORK="${Q4CI_WORK:-${TMPDIR:-/tmp}}"
BIN="$WORK/q4complete-evalci"
MODEL="$WORK/q4complete-evalci-model.bin"
REPORT="$WORK/q4complete-evalci-report.json"
BASELINE=testdata/eval-baseline.json

go build -o "$BIN" ./cmd/q4complete

# Train on the engine's own source. The holdout lives in testdata/ so
# it is never part of the training set.
Q4COMPLETE_MODEL="$MODEL" "$BIN" index -root cmd -root internal -o "$MODEL"

if [ "$1" = "--write-baseline" ]; then
    Q4COMPLETE_MODEL="$MODEL" "$BIN" eval -root testdata/holdout \
        -files 8 -pos 16 -seed 1 -o "$BASELINE"
    echo "wrote baseline $BASELINE"
    exit 0
fi

Q4COMPLETE_MODEL="$MODEL" "$BIN" eval -root testdata/holdout \
    -files 8 -pos 16 -seed 1 -o "$REPORT" \
    -baseline "$BASELINE" -tol 0.15
