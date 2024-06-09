# Transformer_example

This is repo which implements a simple Transformer. It takes the attention mechanism and other network components that Andrej Karpathy developed in [nanoGPT](https://github.com/karpathy/nanoGPT) for language generation and reuses them to implement a Transformer for language translation. It uses the PyTorch framework. The language translation Transformer’s structure follows the example in François Chollet’s book [‘Deep Learning with Python’](https://github.com/fchollet/deep-learning-with-python-notebooks)

The code is talked through in more detail in this post [here](https://johnardavies.github.io/technical/transformer3/). The model is specified in the ```config.py``` and translates from german to english (lower case).

### To run the example:

### 1.  Obtain the code and the data
Clone the repo, cd into the directory and download the english-german translation pairs from [anki](https://www.manythings.org/anki/) and unzip the file:
```
$ wget https://www.manythings.org/anki/deu-eng.zip && unzip deu-eng.zip
```
### 2.  Install the project dependencies
Create and activate virtual environment, here called pytorch_env and install the requirements:
```
$ python3 -m  venv pytorch_env && source pytorch_env/bin/activate &&  pip install -r requirements.txt
```
### 3.  Process the text data
Generate a test and training dataset of the downloaded data:
```
(pytorch_env) $ python text_processing.py
```
### 4.  Train the model
To train a new_model run the train.py script with the "new_model" flag which indicates that the training starts from scratch.
```
(pytorch_env) $ python train.py "new_model"
```
To resume training from a model saved after an earlier training run, saved in the format model_epoch_22, pass it as an argument like:
```
(pytorch_env) $ python train.py model_epoch_22
```
Training the model for 30 epochs on a Mac Air M2 with 24 GB RAM doing other things took about 7.5 days.

### 5.  Generate translations
Use a trained model specified by model_name and an input sentence to generate a translation. If there is no input sentence the model will generate translations from the test dataset.
```
(pytorch_env) $ python translate.py model_name "das wetter ist gut"
```
