# Multiclass neural network classifier

There are few ways to use this.

---------------

## 1) With creation of a temporal ***dump*** file:

#### Create bottleneck dump:
```
python3 train_full_model.py -mb
```
- dump.gz and labels.txt will be created.

#### Train the last layer:
```
python3 train_last_layer.py
```
- weights of the last layer will be stored in "saved_model" directory.

#### Create evaluation pb-model without stop_gradient nodes and so on:
```
python3 train_last_layer.py -ev
```

### or you can continue train full network:
```
python3 train_full_model.py -llr
```
- then only weights of the last layer will be load, and model will be continue to train.

---------------

## 2) Without



