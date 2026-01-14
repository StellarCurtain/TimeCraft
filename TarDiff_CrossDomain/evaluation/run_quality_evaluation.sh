#!/bin/bash
# Usage: bash run_quality_evaluation.sh <dataset> <ms_value> <alpha>
# Examples:
#   bash run_quality_evaluation.sh nasdaq 20 1e-05
#   bash run_quality_evaluation.sh wafer 20 100
#   bash run_quality_evaluation.sh nasdaq_extreme 50 0.5

DATASET=$1
MS=$2
ALPHA=$3

if [ -z "$DATASET" ] || [ -z "$MS" ] || [ -z "$ALPHA" ]; then
    echo "Usage: bash run_quality_evaluation.sh <dataset> <ms_value> <alpha>"
    exit 1
fi

if [ "$DATASET" != "nasdaq" ] && [ "$DATASET" != "wafer" ] && [ "$DATASET" != "nasdaq_extreme" ]; then
    echo "Error: unsupported dataset '$DATASET', use nasdaq, wafer, or nasdaq_extreme"
    exit 1
fi

DATA_DIR="../TarDiff_CrossDomain/data/processed/${DATASET}"

if [ "$DATASET" == "wafer" ]; then
    REF_FILE="train_tuple.pkl"
else
    REF_FILE="train_tuple_50.pkl"
fi

find_synth_file() {
    local base_pattern="synt_tardiff_noise_rnn_train_guidance_sc"
    local suffix="_ms${MS}k.pkl"
    local alpha_val="$1"
    local alpha_formats=()
    
    if [[ "$alpha_val" =~ ^[0-9]+$ ]]; then
        alpha_formats=("${alpha_val}.0")
    else
        local sci_2digit=$(printf "%.0e" "$alpha_val" | sed 's/e\([+-]\)\([0-9]\)$/e\10\2/')
        local sci_1digit=$(printf "%.0e" "$alpha_val" | sed 's/e\([+-]\)0\([0-9]\)$/e\1\2/')
        local decimal=$(printf "%.10f" "$alpha_val" | sed 's/0*$//' | sed 's/\.$//')
        alpha_formats=("$sci_2digit" "$sci_1digit" "$decimal")
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
    SYNTH_FILE_GUIDANCE="synt_tardiff_noise_rnn_train_no_guidance_ms${MS}k.pkl"
    ALPHA_STR="0"
else
    ALPHA_NORMALIZED=$(find_synth_file "$ALPHA")
    SYNTH_FILE_GUIDANCE="synt_tardiff_noise_rnn_train_guidance_sc${ALPHA_NORMALIZED}_ms${MS}k.pkl"
    ALPHA_STR="${ALPHA_NORMALIZED}"
fi

python ../TarDiff_CrossDomain/evaluation/quality_evaluation.py -r "${DATA_DIR}/${REF_FILE}" -s0 "${DATA_DIR}/synt_tardiff_noise_rnn_train_no_guidance_ms${MS}k.pkl" -s1 "${DATA_DIR}/${SYNTH_FILE_GUIDANCE}" -a "${ALPHA}" -o "quality_results_${DATASET}_ms${MS}k_alpha${ALPHA_STR}"
