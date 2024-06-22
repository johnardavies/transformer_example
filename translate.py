import torch
import torch.nn.functional as Fun
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pickle
import text_processing as text_pro

import Transformer as Tr
from config import TransformerConfig

import sys


# Initialize configuration
config = TransformerConfig()

# Load the english and german dictionaries
with open("german_dictionary.pkl", "rb") as file_ger:
    # Load the pickled object
    german_tokens = pickle.load(file_ger)


with open("english_dictionary.pkl", "rb") as file_en:
    # Load the pickled object
    english_tokens = pickle.load(file_en)

# Reverse the dictionaries to go from the numbers back to the words
# This is used previously in text processing consider refactoring
decode_to_english = {v: k for k, v in english_tokens.items()}

decode_to_german = {v: k for k, v in german_tokens.items()}


# Functions to encode and decode the data


def source_vectorization(x):
    """Converts the German words into numbers"""
    return [
        german_tokens[element]
        for element in x.split()
        if element in german_tokens.keys()
    ]


def target_vectorization(x):
    """Converts the English words into numbers"""
    return [
        english_tokens[element]
        for element in x.split()
        if element in english_tokens.keys()
    ]


test_dataset = torch.load("test_dataset.pt")
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

# Set up the transformer and load the past state
model_predict = Tr.Transformer(config)

# Take the model that the script is going to use to translate from as the first argument in the command line
model = sys.argv[1]

state = torch.load(model)

# Loads the model
model_predict.load_state_dict(state["model_state_dict"])

# Sets the model state to evaluate
model_predict.eval()


def prediction(x, y):
    """This gives the probability distribution over English words for a given German translation"""
    logits = model_predict(x, y)
    logits = logits.squeeze(0)  # Will remove the first dimension if it is set to 0
    # returns a tensor with all specified dimensions of input of size 1 removed - in this case the first dimension
    # The dim = -1 applies softmax over the last dimension of the tensor
    return Fun.softmax(logits, dim=-1)


def decode_sequence(input_sentence):
    """This function generates the translation"""

    # Unsqueezing adds an extra dimension so that the model which was trained on batches can read single sentences
    tokenized_input_sentence = text_pro.tensor_pad(
        source_vectorization(input_sentence)
    )[: config.block_size].unsqueeze(0)

    #  initalises the decoded sentence with [start]
    decoded_sentence = "[start]"

    # Loop through the sentence word by word
    for i in range(0, config.block_size):
        tokenized_target_sentence = text_pro.tensor_pad(
            target_vectorization(decoded_sentence)
        )[: config.block_size].unsqueeze(0)

        # Generate predictions
        predictions = prediction(tokenized_input_sentence, tokenized_target_sentence)

        # The first index in the predictions tensor is the word position in the sentence
        # the second index is the predicted word
        # The .item() extracts the tensor index from the tensor
        sampled_token_index = torch.multinomial(predictions[i, :], num_samples=1).item()

        # Gets the word corresponding to the index
        sampled_token = decode_to_english[sampled_token_index]

        # Appends the word to the predicted translation to date
        decoded_sentence += " " + sampled_token

        # If the predicted token is [en]d stop
        if sampled_token == "[end]":
            break
    return decoded_sentence


def trans(x, lan):
    """This is a function to translate the English and German text from the numeric representations that we have of them in the
    pickle file"""
    results = ""
    for elem in x:
        if elem != 0:
            if lan == "ger":
                results = results + " " + decode_to_german[elem]
            if lan == "eng":
                results = results + " " + decode_to_english[elem]
    return results


# Style class to format the print statements
class style:
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"


# If there are no arguments other than the script itself  model to use evaluate some of the test sample
if len(sys.argv) == 2:

    # Looking at selected sentences from the test data
    for i, elem in enumerate(test_dataloader):
        if i % 3400 == 0:

            print(style.BOLD + "Orginal" + style.END)
            german = trans(elem[0].tolist()[0], "ger")
            print(german)
            print(style.BOLD + "Translation" + style.END)
            print(trans(elem[1].tolist()[0], "eng"))
            print(style.BOLD + "Machine Translation" + style.END)
            print(decode_sequence(german))
            print("\n")

# If there are there is also an additional argument
elif len(sys.argv) == 3:
    print(decode_sequence(sys.argv[2]))
    sys.exit()
