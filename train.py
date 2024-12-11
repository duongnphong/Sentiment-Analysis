import os
import string
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from data import findFiles, readLines
from model import RNN
from utils import categoryFromOutput, randomTrainingExample, timeSince


def dataload():
    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)
    category_lines = {}
    all_categories = []
    for filename in findFiles("data/names/*.txt"):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines
    n_categories = len(all_categories)
    return all_categories, category_lines, n_letters, n_categories


def train(rnn, criterion, learning_rate, category_tensor, line_tensor):
    hidden = rnn.initHidden()

    rnn.zero_grad()

    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    loss = criterion(output, category_tensor)
    loss.backward()

    # Add parameters' gradients to their values, multiplied by learning rate
    for p in rnn.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item()


def evaluation(all_losses):
    plt.figure()
    plt.plot(all_losses)
    plt.show()


def main():
    all_categories, category_lines, n_letters, n_categories = dataload()
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)
    criterion = nn.NLLLoss()
    learning_rate = 0.005

    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []

    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample(
            all_categories, category_lines
        )
        output, loss = train(
            rnn, criterion, learning_rate, category_tensor, line_tensor
        )
        current_loss += loss

        # Print ``iter`` number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output, all_categories)
            correct = "✓" if guess == category else "✗ (%s)" % category
            print(
                "%d %d%% (%s) %.4f %s / %s %s"
                % (
                    iter,
                    iter / n_iters * 100,
                    timeSince(start),
                    loss,
                    line,
                    guess,
                    correct,
                )
            )

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    torch.save(rnn.state_dict(), "trained_rnn_model.pth")

    evaluation(all_losses)


if __name__ == "__main__":
    main()
