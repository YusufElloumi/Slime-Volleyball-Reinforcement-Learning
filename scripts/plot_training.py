import csv
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"matplotlib is required for plotting: {exc}")


def main():
    csv_path = Path("results/naruto_training.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")

    episodes = []
    rewards = []
    wins = []
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            episodes.append(int(row["episode"]))
            rewards.append(float(row["reward"]))
            wins.append(int(row["wins"]))

    if not episodes:
        raise ValueError(f"No training rows found in {csv_path}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    ax1.plot(episodes, rewards, marker="o", linewidth=1.5, color="tab:blue")
    ax1.set_ylabel("Episode Reward")
    ax1.set_title("Naruto training progress")
    ax1.grid(True, alpha=0.3)

    ax2.plot(episodes, wins, marker="s", linewidth=1.5, color="tab:green")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Win Count")
    ax2.set_ylim(-0.1, max(1, max(wins)) + 1)
    ax2.grid(True, alpha=0.3)

    out_path = Path("results/naruto_training_plot.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    print(f"Saved training graph to {out_path}")


if __name__ == "__main__":
    main()
