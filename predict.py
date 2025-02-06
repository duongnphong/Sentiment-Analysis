import argparse

import torch

from data import lineToTensor
from model import RNN
from train import dataload


def evaluate(rnn, line_tensor):
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output


def predict(input_line, rnn, all_categories, n_predictions=3):
    print("\n> %s" % input_line)
    with torch.no_grad():  # Disable gradient computation for inference
        # Convert the input line to a tensor
        output = evaluate(rnn, lineToTensor(input_line))

        # Get top N categories
        topv, topi = output.topk(n_predictions, 1, True)
        predictions = []

        for i in range(n_predictions):
            value = topv[0][i].item()
            category_index = topi[0][i].item()
            print("(%.2f) %s" % (value, all_categories[category_index]))
            predictions.append([value, all_categories[category_index]])

    return predictions


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Predict the category of a name")
    parser.add_argument("name", type=str, help="Input name for prediction")
    parser.add_argument(
        "n_predictions",
        type=int,
        nargs="?",
        default=3,
        help="Number of top predictions (default: 3)",
    )
    args = parser.parse_args()

    # Load model and categories
    all_categories, _, n_letters, n_categories = dataload()  # Adjust if needed
    n_hidden = 128
    rnn = RNN(n_letters, n_hidden, n_categories)

    # Load saved model parameters
    rnn.load_state_dict(torch.load("./trained_rnn_model.pth", weights_only=True))
    rnn.eval()  # Set the model to evaluation mode

    # Run prediction
    predict(args.name, rnn, all_categories, args.n_predictions)
