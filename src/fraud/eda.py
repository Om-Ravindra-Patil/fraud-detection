"""Run the named EDA queries in sql/04_eda.sql, save each result and plot the key ones."""

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fraud.config import load_config  # noqa: E402
from fraud.data import read_named_queries  # noqa: E402

SERIES = "#2a78d6"
INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.set_xlabel(xlabel, color=INK_MUTED)
    ax.set_ylabel(ylabel, color=INK_MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_MUTED)


def plot_daily_rate(df, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(df["day"], df["fraud_rate_pct"], color=SERIES, linewidth=2, marker="o", markersize=4)
    _style(ax, "Daily fraud rate, TRANSFER and CASH_OUT", "Day", "Fraud rate (%)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_daily_volume(df, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(df["day"], df["transactions"], color=SERIES, width=0.75)
    _style(ax, "Daily transaction volume, TRANSFER and CASH_OUT", "Day", "Transactions")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_hourly_rate(df, path) -> None:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(df["hour_of_day"], df["fraud_rate_pct"], color=SERIES, width=0.75)
    _style(ax, "Fraud rate by hour of day", "Hour", "Fraud rate (%)")
    ax.set_xticks(range(0, 24, 2))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    cfg = load_config()
    paths = cfg["paths"]
    paths["eda_dir"].mkdir(parents=True, exist_ok=True)
    paths["figures_dir"].mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(paths["duckdb"]), read_only=True)
    results = {}
    for name, query in read_named_queries("04_eda.sql").items():
        df = con.execute(query).df()
        df.to_csv(paths["eda_dir"] / f"{name}.csv", index=False)
        results[name] = df
        print(f"\n== {name} ==\n{df.to_string(index=False)}")

    plot_daily_rate(results["fraud_by_day"], paths["figures_dir"] / "eda_daily_fraud_rate.png")
    plot_daily_volume(results["fraud_by_day"], paths["figures_dir"] / "eda_daily_volume.png")
    plot_hourly_rate(results["fraud_by_hour"], paths["figures_dir"] / "eda_hourly_fraud_rate.png")
    print(f"\nSaved CSVs to {paths['eda_dir']} and charts to {paths['figures_dir']}")


if __name__ == "__main__":
    main()
