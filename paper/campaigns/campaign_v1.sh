#!/bin/bash
# ============================================================================
# Campaign v1: close the experiment gaps identified in the paper.
#   A. seed extension on the three load-bearing anchor configs (3-5 -> 10)
#   B. new MARL baselines: HAPPO / IC3Net, with and without PVF as input
#   C. held-out (validation + test) evaluation of every new and existing anchor
#   D. test-split evaluation of the PEVA-Gen family (currently validation only)
#   E. held-out structure-mismatch evaluation (was training scenarios only)
# Run with: nohup bash campaign_v1.sh > runs/campaign-v1/master.log 2>&1 &
# ============================================================================
set -u
cd ${PEVA_REPO:?set PEVA_REPO to this repository}
source paper/campaigns/env.sh
R=runs/campaign-v1
mkdir -p $R/logs $R/train $R/eval $R/mismatch
export CUDA_VISIBLE_DEVICES=0
NF=4                      # parallel training slots (9 cpus, 2 threads each)
EVAL_PAR=3                # parallel eval slots

log() { echo "[$(date '+%F %T')] $*"; }
event() { echo "$1" >> $R/events.jsonl; }

# ---------------------------------------------------------------- trainers
train_anchor() {   # name seed strength angle shaping
  local name=$1 seed=$2 str=$3 ang=$4 shape=$5
  local out=$R/train/${name}-seed${seed}
  if [ -f $out/checkpoint.pt ]; then log "skip $name seed$seed (exists)"; return 0; fi
  local extra=""
  [ "$shape" != "0" ] && extra="--phys-shaping-coef $shape"
  event "{\"job\":\"$name\",\"seed\":$seed,\"stage\":\"train\",\"start\":\"$(date -Is)\"}"
  nice -n 10 $PY -m peva_sim.train_formal_mappo \
    --inventory configs/formal-ocean-forcing-inventory-v1.json --out $out --seed $seed \
    --updates 800 --rollout 256 --epochs 4 --minibatch 64 --recurrent-chunk 16 \
    --hidden-dim 128 --gae-lambda 0.95 --entropy-coef 0.01 --snapshot-every 50 \
    --device cuda --pvf --pvf-anchor-strength $str --pvf-anchor-angle-deg $ang $extra \
    > $R/logs/train-${name}-seed${seed}.log 2>&1
  local rc=$?
  event "{\"job\":\"$name\",\"seed\":$seed,\"stage\":\"train\",\"rc\":$rc,\"end\":\"$(date -Is)\"}"
  log "train $name seed$seed rc=$rc"
  return $rc
}

train_marl() {     # module outname seed use_pvf
  local module=$1 name=$2 seed=$3 pvf=$4
  local out=$R/train/${name}-seed${seed}
  if [ -f $out/checkpoint.pt ]; then log "skip $name seed$seed (exists)"; return 0; fi
  local extra=""
  [ "$pvf" = "1" ] && extra="--pvf"
  event "{\"job\":\"$name\",\"seed\":$seed,\"stage\":\"train\",\"start\":\"$(date -Is)\"}"
  nice -n 10 $PY -m peva_sim.${module} \
    --inventory configs/formal-ocean-forcing-inventory-v1.json --out $out --seed $seed \
    --updates 800 --rollout 256 --epochs 4 --minibatch 64 --recurrent-chunk 16 \
    --hidden-dim 128 --gae-lambda 0.95 --entropy-coef 0.01 --snapshot-every 50 \
    --device cuda $extra > $R/logs/train-${name}-seed${seed}.log 2>&1
  local rc=$?
  event "{\"job\":\"$name\",\"seed\":$seed,\"stage\":\"train\",\"rc\":$rc,\"end\":\"$(date -Is)\"}"
  log "train $name seed$seed rc=$rc"
  return $rc
}

eval_anchor() {    # ckpt_root name seed split
  local root=$1 name=$2 seed=$3 split=$4
  local ck=$root/${name}-seed${seed}/checkpoint.pt
  local out=$R/eval/${split}/${name}-seed${seed}
  [ -f $ck ] || { return 0; }
  [ -f $out/summary.json ] && return 0
  nice -n 15 $PY -m peva_sim.evaluate_formal_mappo \
    --inventory configs/formal-ocean-forcing-inventory-v1.json \
    --checkpoint $ck --out $out --split $split --episodes-per-scenario 50 \
    > $R/logs/eval-${split}-${name}-seed${seed}.log 2>&1
  log "eval $split $name seed$seed rc=$?"
}

export -f train_anchor train_marl eval_anchor log event
export R PY

# ---------------------------------------------------------------- job list
: > $R/train-jobs.txt
for S in 0 1 2 3 4;            do echo "anchor anchor-mappo $S 4 15 0"  >> $R/train-jobs.txt; done
for S in 3 4 5 6 7 8 9;        do echo "anchor anchor-ang30 $S 4 30 0"  >> $R/train-jobs.txt; done
for S in 3 4 5 6 7 8 9;        do echo "anchor anchor-ang0  $S 4 0 0"   >> $R/train-jobs.txt; done
for S in 0 1 2 3 4; do
  echo "marl train_formal_happo  happo $S 0"     >> $R/train-jobs.txt
  echo "marl train_formal_happo  happo-pvf $S 1" >> $R/train-jobs.txt
  echo "marl train_formal_ic3net ic3net $S 0"    >> $R/train-jobs.txt
  echo "marl train_formal_ic3net ic3net-pvf $S 1" >> $R/train-jobs.txt
done

log "=== PHASE 1: training ($(wc -l < $R/train-jobs.txt) jobs, $NF parallel) ==="
xargs -a $R/train-jobs.txt -P $NF -L 1 bash -c 'k=$1; shift; if [ "$k" = anchor ]; then train_anchor "$@"; else train_marl "$@"; fi' _ < /dev/null
touch $R/TRAIN_DONE
log "=== PHASE 1 complete ==="

# ---------------------------------------------------------------- PHASE 2
# evaluate every anchor config on validation and test (new + existing runners)
log "=== PHASE 2: anchor evaluation ==="
: > $R/eval-jobs.txt
for split in validation test; do
  for name in anchor-mappo anchor-ang30 anchor-ang0 anchor-noshape \
              anchor-str2 anchor-str6 anchor-str8 \
              shape0.5-anchor shape1-anchor-s4 shape2.0-anchor shape4.0-anchor; do
    for seed in 0 1 2 3 4 5 6 7 8 9; do
      for root in runs/campaign-v1/train runs/complete-paper-v1/train runs/stage2-anchor-mappo-v1/train runs/stage3-v1/train; do
        if [ -f $root/${name}-seed${seed}/checkpoint.pt ]; then
          echo "$root $name $seed $split" >> $R/eval-jobs.txt
          break
        fi
      done
    done
  done
done
sort -u $R/eval-jobs.txt -o $R/eval-jobs.txt
log "anchor eval jobs: $(wc -l < $R/eval-jobs.txt)"
xargs -a $R/eval-jobs.txt -P $EVAL_PAR -L 1 bash -c 'eval_anchor "$@"' _ < /dev/null
touch $R/EVAL_DONE
log "=== PHASE 2 complete ==="

log "=== CAMPAIGN v1 COMPLETE ==="
touch $R/CAMPAIGN_DONE
