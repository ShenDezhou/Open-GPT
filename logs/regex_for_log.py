import re

def get_train_loss(file_name):
    pat=re.compile('iter (\d+): loss ([.\d]+),')
    train_loss = []
    with open(file_name,'r') as f:
        for line in f:
            match = pat.match(line)
            print(match.group(1), match.group(2))
            train_loss.append((match.group(1), match.group(2)))
    return train_loss

def get_both_loss():
    adam_loss = get_train_loss("adam.log")
    sophia_loss = get_train_loss("sophia.log")
    return adam_loss, sophia_loss
