# Transformer_example

This is repo which implements a simple Transformer. It takes the attention mechanism and other network components that Andrej Karpathy developed in [nanoGPT](https://github.com/karpathy/nanoGPT) for language generation and reuses them to implement a Transformer for language translation. It uses the PyTorch framework. The language translation Transformer’s structure follows the example in François Chollet’s book [‘Deep Learning with Python’](https://github.com/fchollet/deep-learning-with-python-notebooks)

The code is talked through in more detail in this post [here](https://johnardavies.github.io/technical/transformer3/). The model is specified in the ```config.py``` and translates from german to english (lower case).

To run the example:

1. clone the repo, cd into the directory and download the english-german translation pairs from [anki](https://www.manythings.org/anki/) and unzip the file:
```
$ wget https://www.manythings.org/anki/deu-eng.zip && unzip deu-eng.zip
```
2. create and activate virtual environment, here called pytorch_env
```
$ python3 -m  venv pytorch_env && source pytorch_env/bin/activate
```
3. install the requirements
```
(pytorch_env) $ pip install -r requirements.txt
```
4. process the text data to generate a text and training dataset
```
(pytorch_env) $ python text_processing.py
```
5. train the model 
```
(pytorch_env) $ python train.py
```
6. generate translations from a trained model specified by model_name and an input sentence. If there is no input sentence the model will generate translations from the test dataset.
```
(pytorch_env) $ python translate.py model_name "das wetter ist gut"
```
