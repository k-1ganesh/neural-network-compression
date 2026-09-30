import os
import pandas as pd
import matplotlib.pyplot as plt


RESULT_PATH = "results/compression_results.csv"
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

    # SVD and CP use a single rank.

    for method in ["SVD", "CP"]:

        subset = df[df["method"] == method].dropna(subset=["rank"])

        if len(subset) == 0:

            continue

        plt.plot(subset["rank"],subset["test_accuracy"] * 100,marker="o",label=method,)


    # Tucker uses (r1, r2, r3).
    # In this experiment r1 = r2, so r1 is used
    # as the x-axis and different r3 values are
    # shown as separate curves.

    tucker = df[df["method"] == "Tucker"].dropna(subset=["r1","r3","test_accuracy"])

    for r3 in sorted(tucker["r3"].unique()):

        subset = tucker[tucker["r3"] == r3].sort_values("r1")

        plt.plot(subset["r1"],subset["test_accuracy"] * 100,marker="o",label=f"Tucker (r3={int(r3)})",)


    plt.xlabel("Rank / Tucker r1 (= r2)")

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

    # SVD and CP

    for method in ["SVD", "CP"]:

        subset = df[df["method"] == method].dropna(subset=["rank","reconstruction_error"])

        if len(subset) == 0:

            continue

        plt.plot(subset["rank"],subset["reconstruction_error"],marker="o",label=method,)


    # Tucker

    tucker = df[df["method"] == "Tucker"].dropna(subset=["r1","r3","reconstruction_error"])

    for r3 in sorted(tucker["r3"].unique()):

        subset = tucker[tucker["r3"] == r3].sort_values("r1")

        plt.plot(subset["r1"],subset["reconstruction_error"],marker="o",label=f"Tucker (r3={int(r3)})",)


    plt.xlabel("Rank / Tucker r1 (= r2)")

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

    # Dense baseline

    dense = df[df["method"] == "Dense"]

    if len(dense) > 0:

        dense_accuracy = dense.iloc[0]["test_accuracy"] * 100

        plt.scatter([1.0],[dense_accuracy],marker="*",s=150,label="Dense",)


    # Compare all three compression methods.

    for method in ["SVD", "CP", "Tucker"]:

        subset = df[df["method"] == method].dropna(subset=["compression_ratio_total","test_accuracy"])

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

    # Compare all three methods.

    for method in ["SVD", "CP", "Tucker"]:

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