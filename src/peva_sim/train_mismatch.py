"""Train on mismatch scenarios by delegating to train_formal_mappo."""
import sys
import argparse

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inventory", required=True)
    p.add_argument("--n-targets", type=int, required=True, choices=[2, 5])
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--updates", type=int, default=800)
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=64)
    p.add_argument("--recurrent-chunk", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--entropy-coef", type=float, default=0.01)
    p.add_argument("--snapshot-every", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--pvf", action="store_true")
    p.add_argument("--pvf-anchor-strength", type=float, default=0.0)
    p.add_argument("--pvf-anchor-angle-deg", type=float, default=15.0)
    p.add_argument("--phys-shaping-coef", type=float, default=0.0)
    a = p.parse_args()
    
    if a.pvf_anchor_strength > 0 and not a.pvf:
        p.error("--pvf-anchor-strength requires --pvf")
    
    # Convert to train_formal_mappo arguments
    sys.argv = [
        "train_formal_mappo",
        "--inventory", a.inventory,
        "--mismatch-targets", str(a.n_targets),
        "--out", a.out,
        "--seed", str(a.seed),
        "--updates", str(a.updates),
        "--rollout", str(a.rollout),
        "--epochs", str(a.epochs),
        "--minibatch", str(a.minibatch),
        "--recurrent-chunk", str(a.recurrent_chunk),
        "--hidden-dim", str(a.hidden_dim),
        "--gae-lambda", str(a.gae_lambda),
        "--gamma", str(a.gamma),
        "--entropy-coef", str(a.entropy_coef),
        "--snapshot-every", str(a.snapshot_every),
        "--device", a.device,
    ]
    
    if a.pvf:
        sys.argv.append("--pvf")
    if a.pvf_anchor_strength > 0:
        sys.argv.extend(["--pvf-anchor-strength", str(a.pvf_anchor_strength)])
    if a.pvf_anchor_angle_deg != 15.0:
        sys.argv.extend(["--pvf-anchor-angle-deg", str(a.pvf_anchor_angle_deg)])
    if a.phys_shaping_coef > 0:
        sys.argv.extend(["--phys-shaping-coef", str(a.phys_shaping_coef)])
    
    # Import and run the real trainer
    from .train_formal_mappo import main as train_main
    train_main()

if __name__ == "__main__":
    main()
