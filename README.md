# Multiclass neural network classifier

There are few ways to use this.

1) With creation of a temporal ***dump*** file:
=============
#### Create bottleneck dump:

python3 train_full_model.py -mb

#### Train the last layer:
```python
python3 train_last_layer.py
```

- weights of the last layer will be stored in "saved_model" directory.

#### Create evaluation pb-model without stop_gradient nodes and so on:

python3 train_last_layer.py -ev

2) Without
=======

