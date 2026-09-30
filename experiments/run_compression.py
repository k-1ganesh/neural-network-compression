import os
import time
import pandas as pd
import torch

from data import get_dataloaders
from model import (MLP, get_compressible_weights, replace_compressible_weights)
from evaluate import (evaluate_model, count_parameters)
from utils import set_seed

from compression.svd import compress_svd
from compression.cp import compress_cp
from compression.tucker import compress_tucker


# Configuration
SEED = 42
CHECKPOINT_PATH = "checkpoints/dense_final.pt"
RESULT_PATH = "results/compression_results.csv"

SVD_RANKS = [1,2,4,8,16,32,64,128,256,384,512]
CP_RANKS = [1,2,4,8,16,32,64,128,256]
TUCKER_RANKS = [(4,4,1),(4,4,2),(4,4,4),(8,8,1),(8,8,2),(8,8,4),
                (16,16,1),(16,16,2),(16,16,4),(32,32,1),(32,32,2),(32,32,4),
                (64,64,1),(64,64,2),(64,64,4)]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}\nRun train.py first.")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    return checkpoint


def create_model_from_checkpoint(checkpoint):
    model = MLP().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def evaluate_compressed_model(checkpoint, reconstructed_weights, test_loader):
    model = create_model_from_checkpoint(checkpoint)
    replace_compressible_weights(model, reconstructed_weights)
    metrics = evaluate_model(model, test_loader, DEVICE)
    return metrics["accuracy"]


# Main Experiment
def run_experiment():

    set_seed(SEED)
    os.makedirs("results", exist_ok=True)

    print("=" * 70)
    print("Global SVD + CP + Tucker Compression Experiment")
    print("=" * 70)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    checkpoint = load_checkpoint()

    dense_accuracy = checkpoint["test_accuracy"]
    print(f"\nDense checkpoint accuracy: {dense_accuracy * 100:.4f}%")

    _, test_loader = get_dataloaders()

    # Load Dense Model
    dense_model = create_model_from_checkpoint(checkpoint)
    dense_total_params = count_parameters(dense_model)

    # Extract Compressible block
    dense_weights = get_compressible_weights(dense_model)
    dense_weights = [W.detach().cpu().float() for W in dense_weights]

    dense_block_weight_params = sum(W.numel() for W in dense_weights)

    print(f"\nCompressible dense weight block: {dense_block_weight_params:,} parameters")
    print(f"Entire dense model: {dense_total_params:,} parameters")

    results = []

    # Dense baseline row
    results.append({
        "method": "Dense",
        "rank": None,
        "r1": None,
        "r2": None,
        "r3": None,
        "reconstruction_error": 0.0,
        "decomposition_time_sec": 0.0,
        "compressed_weight_params": dense_block_weight_params,
        "dense_block_weight_params": dense_block_weight_params,
        "compression_ratio_block": 1.0,
        "compressed_total_params": dense_total_params,
        "dense_total_params": dense_total_params,
        "compression_ratio_total": 1.0,
        "test_accuracy": dense_accuracy,
        "accuracy_drop": 0.0,
        "status": "success",
    })

    # SVD Sweep
    print("\n" + "=" * 70)
    print("SVD SWEEP")
    print("=" * 70)

    for rank in SVD_RANKS:

        print(f"\n[SVD] rank = {rank}")

        start_time = time.perf_counter()

        try:
            compression = compress_svd(dense_weights,rank)
            elapsed = time.perf_counter() - start_time

            reconstructed_weights = compression["weights"]
            compressed_weight_params = compression["compressed_parameter_count"]
            accuracy = evaluate_compressed_model(checkpoint,reconstructed_weights,test_loader)
            compressed_total_params = dense_total_params - dense_block_weight_params + compressed_weight_params

            block_compression_ratio = dense_block_weight_params / compressed_weight_params
            total_compression_ratio = dense_total_params / compressed_total_params

            accuracy_drop = dense_accuracy - accuracy

            results.append({
                "method": "SVD",
                "rank": rank,
                "r1": None,
                "r2": None,
                "r3": None,
                "reconstruction_error": compression["reconstruction_error"],
                "decomposition_time_sec": elapsed,
                "compressed_weight_params": compressed_weight_params,
                "dense_block_weight_params": dense_block_weight_params,
                "compression_ratio_block": block_compression_ratio,
                "compressed_total_params": compressed_total_params,
                "dense_total_params": dense_total_params,
                "compression_ratio_total": total_compression_ratio,
                "test_accuracy": accuracy,
                "accuracy_drop": accuracy_drop,
                "status": "success",
            })

            print(f"  Reconstruction Error: {compression['reconstruction_error']:.6f}")
            print(f"  Compressed Weight Params: {compressed_weight_params:,}")
            print(f"  Block Compression Ratio: {block_compression_ratio:.2f}x")
            print(f"  Total Compression Ratio: {total_compression_ratio:.2f}x")
            print(f"  Accuracy: {accuracy * 100:.4f}%")
            print(f"  Accuracy Drop: {accuracy_drop * 100:.4f} percentage points")
            print(f"  Decomposition Time: {elapsed:.3f}s")

        except Exception as exc:

            elapsed = time.perf_counter() - start_time
            print(f"  FAILED: {exc}")

            results.append({
                "method": "SVD",
                "rank": rank,
                "r1": None,
                "r2": None,
                "r3": None,
                "reconstruction_error": None,
                "decomposition_time_sec": elapsed,
                "compressed_weight_params": None,
                "dense_block_weight_params": dense_block_weight_params,
                "compression_ratio_block": None,
                "compressed_total_params": None,
                "dense_total_params": dense_total_params,
                "compression_ratio_total": None,
                "test_accuracy": None,
                "accuracy_drop": None,
                "status": f"failed: {exc}",
            })

    # CP Sweep
    print("\n" + "=" * 70)
    print("CP SWEEP")
    print("=" * 70)

    for rank in CP_RANKS:

        print(f"\n[CP] rank = {rank}")

        start_time = time.perf_counter()

        try:
            compression = compress_cp(dense_weights,rank,n_iter_max=300,tol=1e-6,random_state=SEED)
            elapsed = time.perf_counter() - start_time

            reconstructed_weights = compression["weights"]
            compressed_weight_params = compression["compressed_parameter_count"]
            accuracy = evaluate_compressed_model(checkpoint,reconstructed_weights,test_loader)
            compressed_total_params = dense_total_params - dense_block_weight_params + compressed_weight_params

            block_compression_ratio = dense_block_weight_params / compressed_weight_params
            total_compression_ratio = dense_total_params / compressed_total_params

            accuracy_drop = dense_accuracy - accuracy

            results.append({
                "method": "CP",
                "rank": rank,
                "r1": None,
                "r2": None,
                "r3": None,
                "reconstruction_error": compression["reconstruction_error"],
                "decomposition_time_sec": elapsed,
                "compressed_weight_params": compressed_weight_params,
                "dense_block_weight_params": dense_block_weight_params,
                "compression_ratio_block": block_compression_ratio,
                "compressed_total_params": compressed_total_params,
                "dense_total_params": dense_total_params,
                "compression_ratio_total": total_compression_ratio,
                "test_accuracy": accuracy,
                "accuracy_drop": accuracy_drop,
                "status": "success",
            })

            print(f"  Reconstruction Error: {compression['reconstruction_error']:.6f}")
            print(f"  Compressed Weight Params: {compressed_weight_params:,}")
            print(f"  Block Compression Ratio: {block_compression_ratio:.2f}x")
            print(f"  Total Compression Ratio: {total_compression_ratio:.2f}x")
            print(f"  Accuracy: {accuracy * 100:.4f}%")
            print(f"  Accuracy Drop: {accuracy_drop * 100:.4f} percentage points")
            print(f"  Decomposition Time: {elapsed:.3f}s")

        except Exception as exc:

            elapsed = time.perf_counter() - start_time
            print(f"  FAILED: {exc}")

            results.append({
                "method": "CP",
                "rank": rank,
                "r1": None,
                "r2": None,
                "r3": None,
                "reconstruction_error": None,
                "decomposition_time_sec": elapsed,
                "compressed_weight_params": None,
                "dense_block_weight_params": dense_block_weight_params,
                "compression_ratio_block": None,
                "compressed_total_params": None,
                "dense_total_params": dense_total_params,
                "compression_ratio_total": None,
                "test_accuracy": None,
                "accuracy_drop": None,
                "status": f"failed: {exc}",
            })

    # Tucker Sweep
    print("\n" + "=" * 70)
    print("TUCKER SWEEP")
    print("=" * 70)

    for ranks in TUCKER_RANKS:

        r1, r2, r3 = ranks
        print(f"\n[Tucker] ranks = ({r1},{r2},{r3})")

        start_time = time.perf_counter()

        try:
            compression = compress_tucker(dense_weights,ranks,n_iter_max=100,tol=1e-6,random_state=SEED)

            elapsed = time.perf_counter() - start_time

            reconstructed_weights = compression["weights"]
            compressed_weight_params = compression["compressed_parameter_count"]
            accuracy = evaluate_compressed_model(checkpoint,reconstructed_weights,test_loader)
            compressed_total_params = dense_total_params - dense_block_weight_params + compressed_weight_params

            block_compression_ratio = dense_block_weight_params / compressed_weight_params
            total_compression_ratio = dense_total_params / compressed_total_params

            accuracy_drop = dense_accuracy - accuracy

            results.append({
                "method": "Tucker",
                "rank": None,
                "r1": r1,
                "r2": r2,
                "r3": r3,
                "reconstruction_error": compression["reconstruction_error"],
                "decomposition_time_sec": elapsed,
                "compressed_weight_params": compressed_weight_params,
                "dense_block_weight_params": dense_block_weight_params,
                "compression_ratio_block": block_compression_ratio,
                "compressed_total_params": compressed_total_params,
                "dense_total_params": dense_total_params,
                "compression_ratio_total": total_compression_ratio,
                "test_accuracy": accuracy,
                "accuracy_drop": accuracy_drop,
                "status": "success",
            })

            print(f"  Reconstruction Error: {compression['reconstruction_error']:.6f}")
            print(f"  Compressed Weight Params: {compressed_weight_params:,}")
            print(f"  Block Compression Ratio: {block_compression_ratio:.2f}x")
            print(f"  Total Compression Ratio: {total_compression_ratio:.2f}x")
            print(f"  Accuracy: {accuracy * 100:.4f}%")
            print(f"  Accuracy Drop: {accuracy_drop * 100:.4f} percentage points")
            print(f"  Decomposition Time: {elapsed:.3f}s")

        except Exception as exc:

            elapsed = time.perf_counter() - start_time
            print(f"  FAILED: {exc}")

            results.append({
                "method": "Tucker",
                "rank": None,
                "r1": r1,
                "r2": r2,
                "r3": r3,
                "reconstruction_error": None,
                "decomposition_time_sec": elapsed,
                "compressed_weight_params": None,
                "dense_block_weight_params": dense_block_weight_params,
                "compression_ratio_block": None,
                "compressed_total_params": None,
                "dense_total_params": dense_total_params,
                "compression_ratio_total": None,
                "test_accuracy": None,
                "accuracy_drop": None,
                "status": f"failed: {exc}",
            })

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULT_PATH,index=False)

    print("\n" + "=" * 70)
    print("EXPERIMENT FINISHED")
    print("=" * 70)

    print(f"Results saved to:\n{RESULT_PATH}")

    print("\nSummary:")
    print(
        results_df[
            [
                "method",
                "rank",
                "r1",
                "r2",
                "r3",
                "reconstruction_error",
                "compression_ratio_block",
                "test_accuracy",
                "accuracy_drop",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    run_experiment()