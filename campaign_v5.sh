#!/bin/bash
# campaign-v5 driver.  bash campaign_v5.sh eval|train|status
cd ${PEVA_REPO:?set PEVA_REPO to this repository} || exit 1
. ./env.sh
export PYTHONPATH=src:$PYTHONPATH
R=runs/complete-paper-v3
mkdir -p $R/logs $R/eval $R/train
INV=configs/formal-ocean-forcing-inventory-v1.json
log() { echo "{\"t\":\"$(date -Is)\",$1}" >> $R/events.jsonl; }

run_eval() {  # $1=name, rest=command
  local name=$1; shift
  [ -f $R/eval/$name/summary.json ] && return 0
  rm -rf $R/eval/$name
  nice -n 5 $PY "$@" > $R/logs/eval-$name.log 2>&1
  local rc=$?; log "\"job\":\"$name\",\"kind\":\"eval\",\"rc\":$rc"
}
export -f run_eval log; export R PY

train_std() {  # $1=name $2=seed, rest=extra args
  local N=$1 S=$2; shift 2; local T=$R/train/$N-seed$S
  [ -f $T/DONE ] && return 0
  log "\"job\":\"$N-seed$S\",\"kind\":\"train\",\"start\":true"
  local RES=""; [ -f $T/resume.pt ] && RES="--resume $T/resume.pt"
  nice -n 10 $PY -m peva_sim.train_formal_mappo --inventory $INV --out $T --seed $S --rollout 256 --epochs 4 \
    --minibatch 64 --recurrent-chunk 16 --hidden-dim 128 --gae-lambda 0.95 --entropy-coef 0.01 --device cuda \
    $RES "$@" >> $R/logs/train-$N-seed$S.log 2>&1
  local rc=$?; log "\"job\":\"$N-seed$S\",\"kind\":\"train\",\"rc\":$rc"
  [ $rc -eq 0 ] && touch $T/DONE
}

train_bc() {  # $1=seed
  local S=$1 SK=$R/train/bc-skeleton/checkpoint.pt INIT=$R/train/bc-init-seed$1.pt T=$R/train/bc-finetune-seed$1
  [ -f $T/DONE ] && return 0
  if [ ! -f $SK ]; then  # one tiny update only to obtain a valid checkpoint skeleton (strength 1e-6)
    $PY -m peva_sim.train_formal_mappo --inventory $INV --out $R/train/bc-skeleton --seed 777 --updates 1 \
      --rollout 64 --epochs 1 --minibatch 32 --recurrent-chunk 16 --hidden-dim 128 --device cuda --pvf \
      --pvf-anchor-strength 1e-6 --pvf-anchor-angle-deg 30 > $R/logs/bc-skeleton.log 2>&1 || true
  fi
  [ -f $INIT ] || nice -n 10 $PY tools/bc_control.py pretrain --skeleton $SK --out $INIT --seed $S \
      > $R/logs/bc-pretrain-seed$S.log 2>&1
  log "\"job\":\"bc-pretrain-seed$S\",\"kind\":\"train\",\"rc\":$?"
  # (a) fine-tune the cloned actor directly
  rm -rf $T
  nice -n 10 $PY tools/bc_control.py finetune --init $INIT --warmup-updates 25 --inventory $INV --out $T \
    --seed $S --updates 800 --rollout 256 --epochs 4 --minibatch 64 --recurrent-chunk 16 --hidden-dim 128 \
    --gae-lambda 0.95 --entropy-coef 0.01 --snapshot-every 100 --device cuda --pvf --pvf-anchor-strength 1e-6 \
    --pvf-anchor-angle-deg 30 > $R/logs/bc-finetune-seed$S.log 2>&1
  local rc=$?; log "\"job\":\"bc-finetune-seed$S\",\"kind\":\"train\",\"rc\":$rc"
  [ $rc -eq 0 ] && touch $T/DONE
}
export -f train_std train_bc; export INV

case "$1" in
eval)
  $PY tools/campaign_v5.py
  cut -f1 $R/eval-jobs.tsv > /tmp/v5names; cut -f2 $R/eval-jobs.tsv > /tmp/v5cmds
  paste -d' ' /tmp/v5names /tmp/v5cmds | xargs -P ${2:-3} -L 1 bash -c 'run_eval "$@"' _
  log "\"phase\":\"eval\",\"done\":true"; touch $R/EVAL_DONE ;;
train)
  {
    for S in 3 4 5 6 7 8 9; do echo "train_std shape1-noanchor $S --updates 800 --snapshot-every 50 --pvf --phys-shaping-coef 1.0"; done
    for S in 0 1 2; do echo "train_bc $S"; done
    for S in 0 1 2; do
      echo "train_std long-mappo $S --updates 4000 --snapshot-every 400 --no-communication"
      echo "train_std long-pvf-mappo $S --updates 4000 --snapshot-every 400 --no-communication --pvf"
    done
  } | xargs -P ${2:-6} -I{} bash -c '{}'
  log "\"phase\":\"train\",\"done\":true"; touch $R/TRAIN_DONE ;;
status)
  echo "eval summaries: $(ls $R/eval/*/summary.json 2>/dev/null | wc -l) / $(wc -l < $R/eval-jobs.tsv 2>/dev/null)"
  for d in $R/train/*/; do [ -f $d/updates.jsonl ] && echo "$(basename $d): $(tail -1 $d/updates.jsonl | python3 -c 'import json,sys;r=json.loads(sys.stdin.read());print(r["update"],r["environment_steps"],round(r["elapsed_s"]/3600,2),"h")')"; done
  grep -c '"rc":[1-9]' $R/events.jsonl 2>/dev/null | sed 's/^/failed jobs: /'
  grep '"rc":[1-9]' $R/events.jsonl 2>/dev/null | tail -5
  nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader; uptime ;;
esac
