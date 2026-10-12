#!/bin/bash
# Campaign J: fill Table 4 (ablation). Trained anchor + residual at the angles / strengths that had no
# trained policy: theta in {10,20,40,45,60,90} (alpha=4) and alpha in {1,12} (theta=15); 3 seeds each.
# Same trainer and hyper-parameters as campaign-v1; eval on validation and test, 50 episodes/scenario.
# Run with: nohup bash campaign_j.sh > runs/campaign-j/master.log 2>&1 &
set -u
cd ${PEVA_REPO:?set PEVA_REPO to this repository}
source paper/campaigns/env.sh
R=runs/campaign-j
mkdir -p $R/logs $R/train $R/eval
export CUDA_VISIBLE_DEVICES=0
log() { echo "[$(date '+%F %T')] $*"; }

train_eval() {   # name seed strength angle
  local name=$1 seed=$2 str=$3 ang=$4
  local out=$R/train/${name}-seed${seed}
  if [ ! -f $out/checkpoint.pt ]; then
    nice -n 10 $PY -m peva_sim.train_formal_mappo \
      --inventory configs/formal-ocean-forcing-inventory-v1.json --out $out --seed $seed \
      --updates 800 --rollout 256 --epochs 4 --minibatch 64 --recurrent-chunk 16 \
      --hidden-dim 128 --gae-lambda 0.95 --entropy-coef 0.01 --snapshot-every 50 \
      --device cuda --pvf --pvf-anchor-strength $str --pvf-anchor-angle-deg $ang \
      > $R/logs/train-${name}-seed${seed}.log 2>&1
    log "train $name seed$seed rc=$?"
  fi
  for split in validation test; do
    local ev=$R/eval/${split}/${name}-seed${seed}
    [ -f $ev/summary.json ] && continue
    nice -n 15 $PY -m peva_sim.evaluate_formal_mappo \
      --inventory configs/formal-ocean-forcing-inventory-v1.json \
      --checkpoint $out/checkpoint.pt --out $ev --split $split --episodes-per-scenario 50 \
      > $R/logs/eval-${split}-${name}-seed${seed}.log 2>&1
    log "eval $split $name seed$seed rc=$?"
  done
}
export -f train_eval log
export R PY

: > $R/jobs.txt
for S in 0 1 2; do
  for A in 45 60 90; do echo "anchor-ang$A $S 4 $A" >> $R/jobs.txt; done
  echo "anchor-str1 $S 1 15" >> $R/jobs.txt
  echo "anchor-str12 $S 12 15" >> $R/jobs.txt
done
log "=== campaign J: $(wc -l < $R/jobs.txt) jobs, 4 parallel ==="
xargs -a $R/jobs.txt -P 4 -L 1 bash -c 'train_eval "$@"' _ < /dev/null
log "=== campaign J complete ==="
touch $R/DONE
