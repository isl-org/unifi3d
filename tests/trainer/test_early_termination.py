from unifi3d.trainers.acc_trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt
import pytest


@pytest.fixture
def trainer():
    min_epochs = 1000
    max_epochs = 10000
    return Trainer(min_epochs=min_epochs, max_epochs=max_epochs)


def test_early_stopping(trainer):
    # Generate synthetic validation loss data
    epochs = 10000
    np.random.seed(0)
    val_losses = np.linspace(1.0, 0.1, epochs) * np.exp(
        -np.linspace(0, 4, epochs)
    ) + np.exp(-np.linspace(0, 4, epochs)) * np.random.normal(0, 0.1, epochs)

    filtered_losses = []
    delta_losses = []
    termination_signals = []
    old_loss = trainer.early_terminate["val_loss_old_f"]
    for epoch in range(epochs):
        termination = trainer._epoch_should_terminate_early(val_losses[epoch], epoch)
        filtered_losses.append(trainer.early_terminate["val_loss_old_f"])
        delta_losses.append(old_loss - trainer.early_terminate["val_loss_old_f"])
        old_loss = trainer.early_terminate["val_loss_old_f"]
        termination_signals.append(termination)

    # Plot results
    plt.figure(figsize=(12, 8))
    plt.semilogy(val_losses, label="Validation Loss")
    plt.semilogy(filtered_losses, label="Filtered Loss")
    plt.semilogy(delta_losses, label="Delta Loss")
    plt.axhline(
        y=trainer.early_terminate["threshold"],
        color="red",
        linestyle="--",
        label="Threshold Delta Loss",
    )
    plt.axvline(
        x=trainer.min_epochs, color="purple", linestyle="--", label="Min Epochs"
    )

    # Fill the area below the termination signal
    plt.semilogy(
        termination_signals, label="Termination Signal", color="magenta", alpha=0.75
    )
    plt.fill_between(range(epochs), 0, termination_signals, color="magenta", alpha=0.25)

    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Early Stopping Test")
    plt.legend()
    plt.grid(True)
    plt.savefig("test_early_stopping_plot.png")
    plt.close()

    # Check if termination signal turns True at some point after min_epochs
    assert not any(
        termination_signals[: trainer.min_epochs]
    ), "Early stopping triggered before min_epochs"
