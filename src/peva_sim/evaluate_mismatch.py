"""Evaluate on mismatch scenarios."""
import argparse,json
from pathlib import Path
import numpy as np,torch
from .mismatch_pool import MismatchTargetsPool
from .formal_mappo import FormalMAPPO,features,exchange_actor_messages

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--inventory",required=True)
    p.add_argument("--n-targets",type=int,required=True,choices=[2,5])
    p.add_argument("--out",required=True)
    p.add_argument("--episodes",type=int,default=100)
    a=p.parse_args()
    
    ck=torch.load(a.checkpoint,map_location="cpu",weights_only=False)
    meta=ck["metadata"]
    
    env=MismatchTargetsPool(a.inventory,a.n_targets)
    cfg=env.cfg
    
    # 从checkpoint推断模型架构
    learned_comm = "actor.message_head.weight" in ck["model"]
    communication_type = ("none" if not learned_comm else
                          "tarmac" if "actor.query_head.weight" in ck["model"]
                          else "commnet")
    anchor_strength = float(meta["args"].get("pvf_anchor_strength", 0.0))
    obs_dim = (ck["model"]["actor.message_head.weight"].shape[1]
               if learned_comm else
               ck["model"]["actor.encoder.0.weight"].shape[1])
    state_dim = ck["model"]["critic.0.weight"].shape[1]
    hidden_dim = ck["model"]["actor.encoder.0.weight"].shape[0]
    
    agent = FormalMAPPO(
        obs_dim, state_dim, hidden_dim,
        learned_comm=learned_comm,
        role_dim=cfg.n_uavs,
        communication_type=communication_type,
        anchor_strength=anchor_strength,
        anchor_angle_deg=meta["args"].get("pvf_anchor_angle_deg", 15.0),
        anchor_targets=cfg.n_targets)
    agent.load_state_dict(ck["model"])
    agent.eval()
    
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    results=[]
    for ep in range(a.episodes):
        obs,info=env.reset(ep*1000)
        done=False; ret=0
        hidden = agent.actor.initial_state(cfg.n_uavs)
        mask = torch.zeros(cfg.n_uavs, 1)
        while not done:
            with torch.no_grad():
                actor_input = torch.as_tensor(
                    features(obs, cfg, meta["args"]["pvf"], anchor_strength > 0),
                    dtype=torch.float32)
                communication_mask = None
                if learned_comm:
                    communication_mask, _ = exchange_actor_messages(
                        agent.actor, actor_input, env)
                mean, _, hidden = agent.actor.step(actor_input, hidden, mask, communication_mask)
                action = mean.numpy()
            obs,r,term,trunc,info=env.step(action)
            mask.fill_(1.0)
            ret+=r; done=term or trunc
        results.append({"episode":ep,"return":ret,"info":info})
        if (ep+1)%10==0:
            print(f"ep{ep+1}/{a.episodes} ret={np.mean([x['return'] for x in results]):.3f}")
    
    env.close()
    
    # 安全地提取指标，使用.get()避免KeyError
    summary={"episodes":a.episodes,
             "mean_return":float(np.mean([x["return"] for x in results]))}
    
    # 尝试提取常见指标
    if results and "info" in results[0]:
        info_keys = results[0]["info"].keys()
        for key in ["all_found_rate", "discovery_rate", "survival_score"]:
            if key in info_keys:
                summary[key] = float(np.mean([x["info"][key] for x in results]))
    
    (out/"summary.json").write_text(json.dumps(summary,indent=2))
    with open(out/"episodes.jsonl","w") as f:
        for r in results: f.write(json.dumps(r)+"\n")
    print(json.dumps(summary,indent=2))

if __name__=="__main__":
    main()
