#!/bin/bash
# Campaign K (reviewer round 6). Run: nohup bash campaign_k.sh > runs/campaign-k/master.log 2>&1 &
#  E1 wrong-belief count: belief AND anchor built on Kb=3 while G in {2,5} survivors (eval only, test)
#  E2 OOD for HAPPO / PVF+HAPPO (formal HAPPO evaluator) and PEVA-Gen (its own evaluator, v5 code base)
#  T1 anchor-as-input control: MAPPO whose observation holds u_i and the role (as the anchored actor) but whose
#     action has no anchor term (strength 1e-6), theta=30, seeds 0-4; eval validation/test/OOD
#  T2 BC of the anchor + PPO, seeds 3-9 (same recipe as seeds 0-2); eval validation/test/OOD
# Training waits for campaign J (runs/campaign-j/DONE) so the two campaigns do not compete for the CPU.
set -u
cd ${PEVA_REPO:?set PEVA_REPO to this repository}
source paper/campaigns/env.sh
export PYTHONPATH=src:${PYTHONPATH:-}
R=runs/campaign-k; E4=runs/complete-paper-v4/eval; C3=runs/complete-paper-v3
INV=configs/formal-ocean-forcing-inventory-v1.json
OOD=$C3/ood/ood-inventory.json
V5=${PEVA_BASELINE_ROOT:?}
mkdir -p $R/logs $R/train $E4
log() { echo "[$(date '+%F %T')] $*"; }
export -f log; export R E4 C3 INV OOD V5 PY

ev() {   # name, args... : eval_misspec / eval_beliefcount job, skipped if done
  local name=$1 tool=$2; shift 2
  [ -f $E4/$name/summary.json ] && return 0
  rm -rf $E4/$name
  nice -n 15 $PY tools/$tool "$@" --out $E4/$name > $R/logs/eval-$name.log 2>&1
  log "eval $name rc=$?"
}
export -f ev

ck30() { [ $1 -le 2 ] && echo runs/complete-paper-v1/train/anchor-ang30-seed$1/checkpoint.pt || echo runs/campaign-v1/train/anchor-ang30-seed$1/checkpoint.pt; }
# ---------------------------------------------------------------- guard: Kb == G reproduces eval_misspec
G1=$R/guard; rm -rf $G1; mkdir -p $G1
CK30=runs/complete-paper-v1/train/anchor-ang30-seed0/checkpoint.pt
$PY tools/eval_misspec.py --checkpoint $CK30 --split test --episodes-per-scenario 2 --out $G1/a > /dev/null 2>&1
$PY tools/eval_beliefcount.py --belief-count 3 --checkpoint $CK30 --split test --episodes-per-scenario 2 --out $G1/b > /dev/null 2>&1
$PY - <<'EOF' || { log "GUARD FAILED"; exit 1; }
import json, sys
a = [json.loads(l)["success_rate"] for l in open("runs/campaign-k/guard/a/episodes.jsonl")]
b = [json.loads(l)["success_rate"] for l in open("runs/campaign-k/guard/b/episodes.jsonl")]
print("guard", a, b); sys.exit(0 if a == b else 1)
EOF
log "guard OK"

# ---------------------------------------------------------------- eval jobs (CPU, 2 in parallel)
: > $R/eval-jobs.txt
for G in 2 5; do
  for s in 0 1 2 3 4 5 6 7 8 9; do
    echo "wb-G${G}Kb3-ang30-seed$s eval_beliefcount.py --belief-count 3 --anchor-peaks 3 --true-targets $G --checkpoint $(ck30 $s) --split test --episodes-per-scenario 50" >> $R/eval-jobs.txt
  done
  echo "wb-G${G}Kb3-zero30 eval_beliefcount.py --belief-count 3 --anchor-peaks 3 --true-targets $G --checkpoint runs/complete-paper-v2/train/anchor-zero-residual/checkpoint.pt --zero-residual --angle 30 --split test --episodes-per-scenario 50" >> $R/eval-jobs.txt
  for m in pvf-greedy cbba sector-search aco-search; do
    echo "wb-G${G}Kb3-$m eval_beliefcount.py --belief-count 3 --true-targets $G --classical $m --split test --episodes-per-scenario 50" >> $R/eval-jobs.txt
  done
done
log "=== campaign K: $(wc -l < $R/eval-jobs.txt) wrong-belief evals ==="
xargs -a $R/eval-jobs.txt -P 2 -L 1 bash -c 'ev "$@"' _ < /dev/null &
EVPID=$!

# HAPPO / PVF+HAPPO OOD (formal evaluator), one at a time beside the wrong-belief evals
(
  for s in 0 1 2 3 4; do
    for k in happo happo-pvf; do
      o=$E4/ood-$k-seed$s; [ -f $o/summary.json ] && continue; rm -rf $o
      nice -n 5 $PY -m peva_sim.evaluate_formal_happo --checkpoint runs/campaign-v1/train/$k-seed$s/checkpoint.pt \
        --inventory $INV --scenario-inventory $OOD --split ood_test --episodes-per-scenario 10 --out $o \
        > $R/logs/ood-$k-seed$s.log 2>&1
      log "eval ood-$k-seed$s rc=$?"
    done
  done
  # PEVA-Gen OOD: its own code base; the OOD inventory is copied under that root (the evaluator requires it)
  mkdir -p $V5/configs; cp $OOD $V5/configs/ood-inventory-r6.json
  for s in 0 1 2 3 4; do
    o=runs/peva-ood-r6/peva-gen-seed$s; [ -f $V5/$o/summary.json ] && continue; rm -rf $V5/$o
    (cd $V5 && nice -n 5 .venv/bin/python -m peva_sim.evaluate_formal_peva \
      --checkpoint runs/eaai-final-v5b-peva-gen-seeds0-9/train/peva-gen-seed$s/checkpoint.pt \
      --transport-aux runs/eaai-final-v5b-transport-aux/checkpoint.pt --inventory configs/formal-ocean-forcing-inventory-v1.json \
      --scenario-inventory configs/ood-inventory-r6.json --split ood_test --episodes-per-scenario 10 --device cpu \
      --out $o) > $R/logs/ood-peva-gen-seed$s.log 2>&1
    log "eval ood-peva-gen-seed$s rc=$?"
  done
) &
OPID=$!

# ---------------------------------------------------------------- training (after campaign J)
while [ ! -f runs/campaign-j/DONE ]; do sleep 300; done
log "campaign J done; starting training"
SK=$C3/train/bc-skeleton/checkpoint.pt
train_job() {   # kind seed
  local kind=$1 S=$2
  if [ $kind = ainput ]; then
    local T=$R/train/anchor-input-seed$S
    if [ ! -f $T/checkpoint.pt ]; then
      nice -n 10 $PY -m peva_sim.train_formal_mappo --inventory $INV --out $T --seed $S --updates 800 --rollout 256 \
        --epochs 4 --minibatch 64 --recurrent-chunk 16 --hidden-dim 128 --gae-lambda 0.95 --entropy-coef 0.01 \
        --snapshot-every 100 --device cuda --pvf --pvf-anchor-strength 1e-6 --pvf-anchor-angle-deg 30 \
        > $R/logs/train-anchor-input-seed$S.log 2>&1
      log "train anchor-input seed$S rc=$?"
    fi
    local CKP=$T/checkpoint.pt N=ainput
  else
    local INIT=$C3/train/bc-init-seed$S.pt T=$C3/train/bc-finetune-seed$S
    [ -f $INIT ] || { nice -n 10 $PY tools/bc_control.py pretrain --skeleton runs/complete-paper-v3/train/bc-skeleton/checkpoint.pt \
        --out $INIT --seed $S > $R/logs/bc-pretrain-seed$S.log 2>&1; log "bc pretrain seed$S rc=$?"; }
    if [ ! -f $T/DONE ]; then
      rm -rf $T
      nice -n 10 $PY tools/bc_control.py finetune --init $INIT --warmup-updates 25 --inventory $INV --out $T \
        --seed $S --updates 800 --rollout 256 --epochs 4 --minibatch 64 --recurrent-chunk 16 --hidden-dim 128 \
        --gae-lambda 0.95 --entropy-coef 0.01 --snapshot-every 100 --device cuda --pvf --pvf-anchor-strength 1e-6 \
        --pvf-anchor-angle-deg 30 > $R/logs/bc-finetune-seed$S.log 2>&1 && touch $T/DONE
      log "bc finetune seed$S rc=$?"
    fi
    ev test-bcinit-seed$S eval_misspec.py --checkpoint $INIT
    ev validation-bcinit-seed$S eval_misspec.py --checkpoint $INIT --split validation
    ev ood-bcinit-seed$S eval_misspec.py --checkpoint $INIT --scenario-inventory $OOD --split ood_test --episodes-per-scenario 10
    local CKP=$T/checkpoint.pt N=bcft
  fi
  ev test-$N-seed$S eval_misspec.py --checkpoint $CKP
  ev validation-$N-seed$S eval_misspec.py --checkpoint $CKP --split validation
  ev ood-$N-seed$S eval_misspec.py --checkpoint $CKP --scenario-inventory $OOD --split ood_test --episodes-per-scenario 10
}
export -f train_job; export SK
{ for S in 0 1 2 3 4; do echo "ainput $S"; done; for S in 3 4 5 6 7 8 9; do echo "bc $S"; done; } > $R/train-jobs.txt
xargs -a $R/train-jobs.txt -P 4 -L 1 bash -c 'train_job "$@"' _ < /dev/null
wait $EVPID $OPID
log "=== campaign K complete ==="
touch $R/DONE
