#!/bin/bash
# HMAG runs with the HMAGConfig defaults (L1/L2/L3 = 1,500/2,100/7,500, hot placement, log
# frequency, promotion threshold 0.5, direct promotion to L1, slot indexes). The baseline runs
# are in revision_results/v8; results in revision_results/v9.
cd "$(dirname "$0")"
# run inside a Python environment with the packages of requirements.txt
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 MAX_MS_PER_TOKEN=7.7 REV_MAX_MS_PER_TOKEN=7.7 REV_USE_GPU=1
export SOFT_DEADLINE_EPOCH=$(date -d "+2 days" +%s) HARD_DEADLINE_EPOCH=$(date -d "+2 days" +%s)
V9=revision_results/v9; LOG=$V9/queue.log
echo "$(date) v9 start" >> $LOG
python smoke_hmag.py >> $V9/smoke.log 2>&1 || { echo "$(date) SMOKE FAILED" >> $LOG; exit 1; }
HMAG500_OUT=$V9/hmag500 python run_hmag500.py >> $V9/hmag500/stdout.log 2>&1
echo "$(date) hmag500 exit $?" >> $LOG
MAIN500_OUT=$V9/main500 CONC_METHODS=HMAG CONC_LEVELS=1,2,3,4 CONC_SUFFIX=_c1234 python run_main500.py concurrency >> $V9/main500/stdout_conc.log 2>&1
echo "$(date) concurrency exit $?" >> $LOG
REV_OUT=$V9/r500 REV_NUM_QUERIES=500 REV_SENS_QUERIES=200 REV_CAP_QUERIES=500 \
  python run_revision_experiments.py --stages ablation1 capacity sensitivity >> $V9/r500/stdout.log 2>&1
echo "$(date) ablation/capacity/sensitivity exit $?" >> $LOG
DYN_OUT=$V9/dynamic500 DYN_ONLY=HMAG python run_dynamic500.py >> $V9/dynamic500/stdout.log 2>&1
echo "$(date) dynamic exit $?" >> $LOG
PARA_OUT=$V9/paraphrase500 python run_paraphrase500.py bench >> $V9/paraphrase500/stdout.log 2>&1
echo "$(date) paraphrase exit $?" >> $LOG
for T in 0.3 1.0; do
  HMAG_SEEDS=1 HMAG_TEMPERATURE=$T HMAG500_OUT=$V9/hmag500 python run_hmag500.py >> $V9/hmag500/stdout_temp.log 2>&1
  echo "$(date) temp $T llama32 exit $?" >> $LOG
done
export HMAG_MODEL=qwen3-4b-instruct:latest HMAG_SEEDS=1 MAX_MS_PER_TOKEN=15
HMAG500_OUT=$V9/hmag500 python run_hmag500.py >> $V9/hmag500/stdout_qwen.log 2>&1
echo "$(date) qwen hmag500 exit $?" >> $LOG
for T in 0.3 1.0; do
  HMAG_TEMPERATURE=$T HMAG500_OUT=$V9/hmag500 python run_hmag500.py >> $V9/hmag500/stdout_temp.log 2>&1
  echo "$(date) temp $T qwen exit $?" >> $LOG
done
unset HMAG_MODEL HMAG_SEEDS; export MAX_MS_PER_TOKEN=7.7
STRESS_SEEDS=1 STRESS_ONLY=full_system,hmag_no_admission,hierarchical_lru,hierarchical_lfu,hmag_fixed_temp,no_hierarchical_cache,hmag_original_placement \
  STRESS_OUT=$V9/stress REV_OUT=$V9/stress python run_stress.py >> $V9/stress/stdout.log 2>&1
echo "$(date) stress exit $?" >> $LOG
echo "$(date) V9 HMAG DONE" >> $LOG
