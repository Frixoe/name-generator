# Canadian Name Generator

Generates names using a Recurrent Neural Network created in Keras and trained on a Canadian Names Dataset(provided with the code).

## Note

- I had to remove a few LSTM layers to get it to work properly as the script was using a lot of GPU memory.
- If you have less than 8GBs of GPU memory, do not add any more layers as this will result in a "Segmentation Fault" and your script will not run.
- If you are still getting a "Seg Fault" trying switching Keras's backend between `theano` and `tensorflow` and seeing what works best. `theano` worked for me with one LSTM layer.
