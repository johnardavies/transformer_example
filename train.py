from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as Fun
from torch import nn
import torch.optim as optim


import sys


# Import the model config and the Transformer class
import config
from Transformer import *

writer = SummaryWriter()

model = Transformer(config)


# Import the training and test datasets and convert them into data loaders
train_dataset = torch.load("train_dataset.pt")
test_dataset = torch.load("test_dataset.pt")

train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=True)

torch.set_grad_enabled(True)

optimizer = optim.Adam(model.parameters())


def save_checkpoint(model, optimizer, save_path, epoch):
    """function to save checkpoints on the model weights, the optimiser state and epoch"""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
        },
        save_path,
    )


# If a saved model is input load the model and the optimizer state, else start from 0
if sys.argv[1] == "new_model":
    epoch_start = 0
elif sys.argv[1] is not None:
    state = torch.load(sys.argv[1])
    model.load_state_dict(state["model_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    epoch_start = state["epoch"]


# The training loop (The + 1 is get the numbers we want from range)
for epoch in range(epoch_start + 1, config.epochs + 1):
    model.train()
    for i, (german, english, output) in enumerate(train_dataloader):

        # clear any existing gradients to compute a new pone
        optimizer.zero_grad()

        # Generates a predicted translation
        yhat = model(german, english)

        # yhat has dimensions (batch_size, block_size, vocab_size)
        # yhat.view(-1, yhat.size(-1)) has the dimensions ((batch_size X block_size), vocab_size)
        # output has the dimensions (batch_size, block_size)
        # output.view(-1) has the dimensions of (batch_size x block_size)

        # Calculates the cross entropy
        loss = Fun.cross_entropy(
            yhat.view(-1, yhat.size(-1)), output.view(-1), ignore_index=0
        )
        # write the loss
        writer.add_scalar("Loss/train", loss, epoch)
        loss.backward()

        # update the weights
        optimizer.step()

        # print how the training is doing at regular stages during the epoch
        if i % 400 == 0:
            print(f"Epoch: {epoch}, Iteration: {i}, Loss: {loss.item()}")
        writer.flush()

    # Save the model and optimizer
    path_to_save = "model_post_" + str(epoch)
    save_checkpoint(model, optimizer, path_to_save, epoch)

    # Compute average loss on the test set
    model.eval()  # Set the model to evaluation mode turning off dropout
    val_loss = 0.0
    with torch.no_grad():  # No gradient computation for validation
        for german_test, english_test, output_test in test_dataloader:
            yhat = model(german_test, english_test)
            loss = Fun.cross_entropy(
                yhat.view(-1, yhat.size(-1)), output_test.view(-1), ignore_index=0
            )
            val_loss += loss.item()
        avg_val_loss = val_loss / len(test_dataloader)
        writer.add_scalar("Loss/test", avg_val_loss, epoch)
        print(f"Epoch: {epoch},  Avg_val_loss: {avg_val_loss}")
