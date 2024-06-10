import torch
import torch.nn.functional as Fun
from torch.utils.data.dataset import random_split

from collections import Counter
from string import punctuation
import pickle

from config import TransformerConfig

# Initialize configuration
config = TransformerConfig()

# Strip punctuation
def strip_punctuation(s):
    """Function that removes punctuation"""
    return str("".join(c for c in s if c not in punctuation))


def tensor_pad(x):
    """converts list to tensor and pads to length 20"""
    return Fun.pad(torch.tensor(x, dtype=torch.int64), (0, config.block_size), value=0)


def main():
    """This is the function which processes the text to output it in a form used by the model"""
    text_file = "deu.txt"
    with open(text_file) as language_file:
        lines = language_file.read().split("\n")
        print("There are " + str(len(lines)) + " lines")

    # Split the text file into German and English, setting it to lower case and removing punctuation
    # We append a [start] at the beginning of each line of the English text and end it with [end]
    text_pairs = []
    for i, line in enumerate(lines):
        try:
            english, german = line.split("\t")[0:2]
            english = "[start] " + strip_punctuation(english.lower()) + " [end]"
            text_pairs.append([strip_punctuation(german.lower()), english])
        except:
            print("failed to load line %s" % (i))

    # To get the tokens used in English and German texts we create
    # two separate lists and from these create a single German text and a single English text
    german_list = [item[0] for item in text_pairs]
    english_list = [item[1] for item in text_pairs]

    german_text = " ".join(german_list)
    english_text = " ".join(english_list)

    class WordsNumbers:
        """Class that produces two dictionaries from input text, one of which maps words
        to numbers and the other one reverses this mapping numbers to words."""

        def __init__(self, text, config):
            self.text = text
            self.config = config
            # Get the tokens
            self.tokens_list = list(set(self.text.split()))

            # Get the most common tokens
            # Subtract -1 as we want to leave 1 of the 15,0000 numbers to code the padding token rather than words. We use 0 for this.
            self.tokens_counter = Counter(self.text.split()).most_common(
                self.config.vocab_size - 1
            )
            self.tokens_vocab = [item for item, count in self.tokens_counter]

        def to_words(self):
            # Converts from the numbers to words
            word_dict = {i + 1: token for i, token in enumerate(self.tokens_vocab)}
            return word_dict

        def to_numbers(self):
            # Comverts from words to numbers
            number_dict = {token: i + 1 for i, token in enumerate(self.tokens_vocab)}
            return number_dict

    # Get the German tokens and the English tokens
    english_tokens = WordsNumbers(english_text, config)
    german_tokens = WordsNumbers(german_text, config)

    # Creates dictionaries converting words into numbers
    english_numeric = english_tokens.to_numbers()
    german_numeric = german_tokens.to_numbers()

    # Save the English and German dictionaries
    with open("english_dictionary.pkl", "wb") as eng:
        pickle.dump(english_numeric, eng)

    with open("german_dictionary.pkl", "wb") as ger:
        pickle.dump(german_numeric, ger)

    # Encode the words into numbers if they are in the 15000 most common ones
    text_pairs_encoded = [
        [
            [
                german_numeric[element]
                for element in pair[0].split()
                if element in german_tokens.tokens_vocab
            ],
            [
                english_numeric[element]
                for element in pair[1].split()
                if element in english_tokens.tokens_vocab
            ],
        ]
        for pair in text_pairs
    ]

    # Split the data between:
    # the encoder input in german
    # the encoder input in english, where the end token is removed elem[1][:-1]
    # english output we are trying to predict which is shifted one token to the right elem[1][1:]
    text_pairs_encoded_split = [
        (elem[0], elem[1][:-1], elem[1][1:]) for elem in text_pairs_encoded
    ]

    # Pads each bit of text to 20 tokens by adding 0s with tensor_pad and truncating at block size
    text_pairs_encoded_padded = [
        [
            tensor_pad(item1)[: config.block_size],
            tensor_pad(item2)[: config.block_size],
            tensor_pad(item3)[: config.block_size],
        ]
        for item1, item2, item3 in text_pairs_encoded_split
    ]
    # Calculate how many observations are needed for a 20% test sample
    test_len = round(len(text_pairs_encoded_padded) * 0.2)

    # Calculate the number of training observations as the residual
    train_len = round(len(text_pairs_encoded_padded)) - test_len

    # Get the train dataset and the test dataset
    train_dataset, test_dataset = random_split(
        text_pairs_encoded_padded, [train_len, test_len]
    )

    # Save the test and train datasets as pickle files
    torch.save(train_dataset, "train_dataset.pt")
    torch.save(test_dataset, "test_dataset.pt")


# Only run main if this script is run directly
if __name__ == "__main__":
    main()
