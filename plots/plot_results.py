import os
import pandas as pd
import matplotlib.pyplot as plt

RESULT_PATH = ("results/compression_results.csv")
PLOT_DIR = "plots"

def load_results():
    if not os.path.exists(RESULT_PATH):
        raise FileNotFoundError(f"Results file not found:\n{RESULT_PATH}\n\nRun the compression experiment first.")

    df = pd.read_csv(RESULT_PATH)
    # Keep only successful experiments.
    df = df[df["status"] == "success"].copy()
    return df

def plot_rank_vs_accuracy(df):
    plt.figure(figsize=(8, 6))

    for method in ["SVD", "CP"]:
        subset = df[df["method"] == method].dropna(subset=["rank"])

        if len(subset) == 0:
            continue
        plt.plot(subset["rank"],subset["test_accuracy"] * 100,marker="o",label=method,)

    plt.xlabel("Rank")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Rank vs Test Accuracy")

    plt.xscale("log", base=2)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR,"rank_vs_accuracy.png"),dpi=200)
    plt.close()


def plot_rank_vs_reconstruction_error(df):
    plt.figure(figsize=(8, 6))

    for method in ["SVD", "CP"]:

        subset = df[df["method"] == method].dropna(subset=["rank","reconstruction_error"])
        if len(subset) == 0:
            continue
        plt.plot(subset["rank"],subset["reconstruction_error"],marker="o",label=method,)

    plt.xlabel("Rank")
    plt.ylabel("Relative Frobenius Reconstruction Error")
    plt.title("Rank vs Reconstruction Error")
    plt.xscale("log", base=2)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR,"rank_vs_reconstruction_error.png"),dpi=200)

    plt.close()


def plot_compression_vs_accuracy(df):
    plt.figure(figsize=(8, 6))

    dense = df[df["method"] == "Dense"]

    if len(dense) > 0:
        dense_accuracy = (dense.iloc[0]["test_accuracy"] * 100)
        plt.scatter([1.0],[dense_accuracy],marker="*",s=150,label="Dense",)

    for method in ["SVD", "CP"]:

        subset = df[df["method"] == method].dropna(subset=["compression_ratio_total","test_accuracy",])

        if len(subset) == 0:
            continue
        plt.plot(subset["compression_ratio_total"],subset["test_accuracy"] * 100,marker="o",label=method,)

    plt.xlabel("Overall Compression Ratio (×)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Compression Ratio vs Test Accuracy")

    plt.xscale("log", base=2)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR,"compression_vs_accuracy.png"),dpi=200)

    plt.close()


def plot_compression_vs_reconstruction_error(df):
    plt.figure(figsize=(8, 6))

    for method in ["SVD", "CP"]:

        subset = df[df["method"] == method].dropna(subset=["compression_ratio_block","reconstruction_error"])

        if len(subset) == 0:
            continue

        plt.plot(subset["compression_ratio_block"],subset["reconstruction_error"],marker="o",label=method,)

    plt.xlabel("Compression Ratio of Compressible Block (×)")
    plt.ylabel("Relative Frobenius Reconstruction Error")
    plt.title("Compression Ratio vs Reconstruction Error")

    plt.xscale("log", base=2)

    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()

    plt.savefig(os.path.join(PLOT_DIR,"compression_vs_reconstruction_error.png"),dpi=200)

    plt.close()


def main():

    os.makedirs(PLOT_DIR,exist_ok=True)

    df = load_results()

    plot_rank_vs_accuracy(df)

    plot_rank_vs_reconstruction_error(df)

    plot_compression_vs_accuracy(df)

    plot_compression_vs_reconstruction_error(df)

    print(f"Plots saved to: {PLOT_DIR}/")

if __name__ == "__main__":
    main()