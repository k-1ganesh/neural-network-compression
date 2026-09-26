import torch
import torch.nn as nn
from tqdm import tqdm
import os

from data import get_dataloaders
from model import MLP
from evaluate import evaluate_model, count_parameters
from utils import set_seed

# Configuration 
SEED = 42
EPOCHS = 20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
CHECKPOINT_PATH = "checkpoints/dense_final.pt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Training 
def train():

    set_seed(SEED)
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load Data
    train_loader, test_loader = get_dataloaders()

    # Model Initialization
    model = MLP().to(DEVICE)
    print(f"Total trainable parameters: {count_parameters(model):,}")

    # Loss + optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    # Training Loop
    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0
        total_examples = 0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch:02d}/{EPOCHS}",
        )

        for images, labels in progress_bar:

            images = images.to(DEVICE,non_blocking=True)
            labels = labels.to(DEVICE,non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)

            loss = criterion(logits,labels)

            loss.backward()

            optimizer.step()

            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_examples += batch_size

            progress_bar.set_postfix(loss=running_loss / total_examples)

        epoch_loss = running_loss / total_examples

        print(
            f"Epoch {epoch:02d} | "
            f"Training Loss: {epoch_loss:.6f}"
        )


    # Test Evaluation
    test_metrics = evaluate_model(model,test_loader,DEVICE)
    test_accuracy = test_metrics["accuracy"]

    print("\n" + "=" * 60)
    print("Dense baseline")
    print("=" * 60)

    print(f"Test Loss:     {test_metrics['loss']:.6f}")
    print(f"Test Accuracy: {test_accuracy * 100:.4f}%")
    print(f"Parameters:    {count_parameters(model):,}")


    # Save checkpoint

    # Save tensors on CPU for portability.
    cpu_state_dict = {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
    }

    checkpoint = {
        "model_state_dict": cpu_state_dict,
        "test_accuracy": test_accuracy,
        "epoch": EPOCHS,
        "seed": SEED,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
    }

    torch.save(checkpoint,CHECKPOINT_PATH)

    print(f"\nSaved dense checkpoint to:\n{CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()