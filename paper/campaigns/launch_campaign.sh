#!/bin/bash
# Launcher for campaign v1.
# Key points: detach with setsid, and give the background job its own stdin,
# otherwise the ssh session blocks on the inherited pipe until the job ends.
cd ${PEVA_REPO:?set PEVA_REPO to this repository} || exit 1
mkdir -p runs/campaign-v1/logs

PIDF=runs/campaign-v1/campaign.pid
if [ -f "$PIDF" ] && ps -p "$(cat "$PIDF")" >/dev/null 2>&1; then
  echo "campaign already running with pid $(cat "$PIDF")"
  exit 0
fi

setsid nohup bash campaign_v1.sh </dev/null >runs/campaign-v1/master.log 2>&1 &
echo "$!" > "$PIDF"
echo "launched campaign pid $!"

sleep 20
echo "--- master.log ---"
tail -15 runs/campaign-v1/master.log
echo "--- training procs ---"
pgrep -fc "peva_sim.train" || echo 0
