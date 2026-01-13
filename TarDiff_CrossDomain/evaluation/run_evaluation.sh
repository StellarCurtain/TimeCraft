#!/bin/bash
# Usage: bash run_evaluation.sh <dataset> <ms_value> <alpha> [--skip-baseline]
# Examples:
#   bash run_evaluation.sh nasdaq 20 0
#   bash run_evaluation.sh nasdaq 50 1e-05
#   bash run_evaluation.sh wafer 50 1e-05 --skip-baseline

DATASET=$1
MS=$2
ALPHA=$3
SKIP_BASELINE=false

if [ -z "$DATASET" ] || [ -z "$MS" ] || [ -z "$ALPHA" ]; then
    echo "Usage: bash run_evaluation.sh <dataset> <ms_value> <alpha> [--skip-baseline]"
    exit 1
fi

for arg in "$@"; do
    if [ "$arg" = "--skip-baseline" ] || [ "$arg" = "-sb" ]; then
        SKIP_BASELINE=true
    fi
done

if [ "$DATASET" != "nasdaq" ] && [ "$DATASET" != "wafer" ]; then
    echo "Error: unsupported dataset '$DATASET', use nasdaq or wafer"
    exit 1
fi

DATA_DIR="../TarDiff_CrossDomain/data/processed/${DATASET}"

find_synth_file() {
    local base_pattern="synt_tardiff_noise_rnn_train_guidance_sc"
    local suffix="_ms${MS}k.pkl"
    local alpha_val="$1"
    local alpha_formats=()
    
    if [[ "$alpha_val" =~ ^[0-9]+$ ]]; then
        alpha_formats=("${alpha_val}.0")
    else
        # First try the original value as-is
        local original="$alpha_val"
        # Decimal format (strip trailing zeros)
        local decimal=$(printf "%.10f" "$alpha_val" | sed 's/0*$//' | sed 's/\.$//')
        # Scientific notation formats
        local sci_2digit=$(printf "%.0e" "$alpha_val" | sed 's/e\([+-]\)\([0-9]\)$/e\10\2/')
        local sci_1digit=$(printf "%.0e" "$alpha_val" | sed 's/e\([+-]\)0\([0-9]\)$/e\1\2/')
        # Priority: original input > decimal > scientific notation
        alpha_formats=("$original" "$decimal" "$sci_2digit" "$sci_1digit")
    fi
    
    for fmt in "${alpha_formats[@]}"; do
        local candidate="${DATA_DIR}/${base_pattern}${fmt}${suffix}"
        if [ -f "$candidate" ]; then
            echo "$fmt"
            return 0
        fi
    done
    
    echo "${alpha_formats[0]}"
    return 1
}

if [ "$ALPHA" = "0" ]; then
    SYNTH_FILE="synt_tardiff_noise_rnn_train_no_guidance_ms${MS}k.pkl"
    ALPHA_STR="0"
else
    ALPHA_NORMALIZED=$(find_synth_file "$ALPHA")
    SYNTH_FILE="synt_tardiff_noise_rnn_train_guidance_sc${ALPHA_NORMALIZED}_ms${MS}k.pkl"
    ALPHA_STR="${ALPHA_NORMALIZED}"
fi

if [ "$DATASET" == "wafer" ]; then
    REAL_DATA="${DATA_DIR}/train_tuple.pkl"
else
    REAL_DATA="${DATA_DIR}/train_tuple_50.pkl"
fi

SYNTH_DATA="${DATA_DIR}/${SYNTH_FILE}"
VAL_DATA="${DATA_DIR}/val_tuple.pkl"
TEST_DATA="${DATA_DIR}/test_tuple.pkl"
OUTPUT_DIR="results_${DATASET}_ms${MS}k_alpha${ALPHA_STR}"

if [ ! -f "$SYNTH_DATA" ]; then
    echo "Error: synthetic data file not found: ${SYNTH_DATA}"
    exit 1
fi

if [ "$SKIP_BASELINE" = false ]; then
    python ../TarDiff_CrossDomain/evaluation/run_experiment.py -r "$REAL_DATA" -s "$SYNTH_DATA" -v "$VAL_DATA" -t "$TEST_DATA" --real_ratio 1.0 --synth_ratio 0.0 --name baseline -o "$OUTPUT_DIR"
fi

python ../TarDiff_CrossDomain/evaluation/run_experiment.py -r "$REAL_DATA" -s "$SYNTH_DATA" -v "$VAL_DATA" -t "$TEST_DATA" --real_ratio 0.0 --synth_ratio 1.0 --name TSTR -o "$OUTPUT_DIR"

python ../TarDiff_CrossDomain/evaluation/run_experiment.py -r "$REAL_DATA" -s "$SYNTH_DATA" -v "$VAL_DATA" -t "$TEST_DATA" --real_ratio 1.0 --synth_ratio 1.0 --name TSRTR -o "$OUTPUT_DIR"

python ../TarDiff_CrossDomain/evaluation/run_experiment.py -r "$REAL_DATA" -s "$SYNTH_DATA" -v "$VAL_DATA" -t "$TEST_DATA" --real_ratio 0.5 --synth_ratio 0.5 --name mixed_50_50 -o "$OUTPUT_DIR"
