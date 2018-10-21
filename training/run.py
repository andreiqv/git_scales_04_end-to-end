import os
os.system('python3 train_full_model.py -mb')
os.system('python3 train_last_layer.py > _log_train.txt')
os.system('python3 train_full_model.py -llr')



