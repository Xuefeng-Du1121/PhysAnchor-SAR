#!/bin/bash
# campaign-v4: (1) the "two PPO updates" anchor at ten seeds, with exactly the
# configuration of the single checkpoint reported so far (runs/smoke-anchor-train:
# updates=2, rollout=64, epochs=1, minibatch=32, alpha=4, theta=15);
# (2) a zero-residual anchor (head and role bias set to zero, i.e. pure physics
# policy) built from that same checkpoint. Held-out test and validation evaluation
# with the official evaluator, 50 episodes per scenario (100 per split).
cd ${PEVA_REPO:?set PEVA_REPO to this repository} || exit 1
. ./env.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
R=runs/complete-paper-v2
INV=configs/formal-ocean-forcing-inventory-v1.json
mkdir -p $R/train $R/test $R/validation $R/logs
log() { echo "{\"t\":\"$(date -Is)\",$1}" >> $R/events.jsonl; }

train() {  # seed
  local O=$R/train/anchor-2upd-seed$1
  [ -f $O/checkpoint-step-000000128.pt ] && return 0
  $PY -m peva_sim.train_formal_mappo --inventory $INV --out $O --seed $1 \
     --updates 2 --rollout 64 --epochs 1 --minibatch 32 --pvf \
     --pvf-anchor-strength 4.0 --pvf-anchor-angle-deg 15.0 --snapshot-every 1 \
     --device cpu > $R/logs/train-2upd-seed$1.log 2>&1
}
evaluate() {  # checkpoint name split episodes-per-scenario
  local O=$R/$3/$2
  [ -f $O/summary.json ] && return 0
  [ -d $O ] && mv $O ${O}_partial_$(date +%s)
  $PY -m peva_sim.evaluate_formal_mappo --checkpoint $1 --inventory $INV --out $O \
     --split $3 --episodes-per-scenario $4 > $R/logs/eval-$3-$2.log 2>&1
  local rc=$?
  [ -f $O/summary.json ] && log "\"job\":\"$2\",\"split\":\"$3\",\"rc\":$rc,\"ok\":true" \
                         || log "\"job\":\"$2\",\"split\":\"$3\",\"rc\":$rc,\"ok\":false"
}

# zero-residual checkpoint
ZR=$R/train/anchor-zero-residual/checkpoint.pt
if [ ! -f $ZR ]; then
  mkdir -p $(dirname $ZR)
  $PY - <<'PY'
import torch
src = "runs/smoke-anchor-train/checkpoint-step-000000128.pt"
ck = torch.load(src, map_location="cpu", weights_only=False)
for k in ("actor.head.weight", "actor.head.bias", "actor.role_bias"):
    ck["model"][k] = torch.zeros_like(ck["model"][k])
ck["metadata"]["derived_from"] = src
ck["metadata"]["derivation"] = "zero residual: actor.head and actor.role_bias set to 0 (pure physics anchor)"
torch.save(ck, "runs/complete-paper-v2/train/anchor-zero-residual/checkpoint.pt")
print("zero-residual checkpoint written")
PY
fi

if [ "$1" = "guard" ]; then
  train 100 || exit 1
  G=$R/guard; rm -rf $G
  $PY -m peva_sim.evaluate_formal_mappo --checkpoint $R/train/anchor-2upd-seed100/checkpoint-step-000000128.pt \
     --inventory $INV --out $G/2upd --split test --episodes-per-scenario 1 > $R/logs/guard.log 2>&1
  $PY -m peva_sim.evaluate_formal_mappo --checkpoint $ZR \
     --inventory $INV --out $G/zero --split test --episodes-per-scenario 1 >> $R/logs/guard.log 2>&1
  ls $G/2upd/summary.json $G/zero/summary.json && echo GUARD_OK || { tail -20 $R/logs/guard.log; exit 1; }
  rm -rf $G $R/train/anchor-2upd-seed100
  exit 0
fi

log "\"stage\":\"start\""
for s in 0 1 2 3 4 5 6 7 8 9; do train $s; done
log "\"stage\":\"trained\""
jobs_list=()
for s in 0 1 2 3 4 5 6 7 8 9; do
  jobs_list+=("$R/train/anchor-2upd-seed$s/checkpoint-step-000000128.pt anchor-2upd-seed$s test 50")
done
jobs_list+=("$ZR anchor-zero-residual test 50")
jobs_list+=("$ZR anchor-zero-residual validation 50")
# two workers
i=0
for j in "${jobs_list[@]}"; do
  evaluate $j &
  i=$((i+1))
  if [ $((i % 2)) -eq 0 ]; then wait; fi
done
wait
log "\"stage\":\"done\""
touch $R/CAMPAIGN_V4_DONE
