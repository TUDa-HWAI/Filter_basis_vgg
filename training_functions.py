import torch
import numpy as np
import math
import torch._C

# area ranking

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=5, verbose=False, delta=0.1, path='checkpoint_MNIST.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved. Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement. Default: 0
            path (str): Path for the checkpoint to be saved to. Default: 'checkpoint.pt'
            trace_func (function): trace print function. Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_acc_min = 0
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_acc, model):

        score = val_acc

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_acc, model)

        # If the next change in precision is within the threshold we set, this time the precision is temporarily recorded
        # If this accuracy is maintained for more than five consecutive times, then early stop
        elif score <= self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        '''Saves model when validation accuracy changed apparently.'''
        if self.verbose:
            self.trace_func(
                f'Validation accuracy changed ({self.val_acc_min:.4f}% --> {val_acc:.4f}%).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_acc_min = val_acc


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 2 every 30 epochs"""
    lrn = lr * (0.5 ** ((epoch * 1.1) // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lrn

def train(model, device, train_loader, val_loader, optimizer, epochs, criterion, patience, delta, path,train_scheduler):
    """
       Trains and validates a given model, incorporating early stopping and model pruning.

       Args:
           model: The neural network model to be trained and validated.
           num_areas: The number of selected weights.
           device: The device (CPU or GPU) on which to perform training and validation.
           train_loader: DataLoader for the training dataset.
           val_loader: DataLoader for the validation dataset.
           optimizer: The optimization algorithm used to update the model's weights.
           epochs: The total number of training epochs.
           criterion: The loss function used to evaluate the model's performance.
           patience: The patience for early stopping (number of epochs to wait after last time validation accuracy improved).
           delta: The minimum change in the monitored quantity to qualify as an improvement.
    """
    total_val = 0
    correct = 0
    total_train = 0
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    valid_accu= []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=delta, path=path)

    for epoch in range(1, epochs + 1):

        ###################
        # train the model #
        ###################

        #adjust_learning_rate(optimizer,epoch, lr)
        model.train()  # prep model for training
        for batch_idx, (data, target) in enumerate(train_loader, 1):
            data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            #loss.register_hook(backward_ste)
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

            # print the training process
            total_train += len(data)
            print('\rbatch '+str(batch_idx), end='', flush=True)
        train_scheduler.step()
        progress = math.ceil(batch_idx / len(train_loader) * 50)
        print("\rTrain epoch %d: %d/%d, [%-51s] %d%%" %
              (epoch, total_train, len(train_loader.dataset),
               '-' * progress + '>', progress * 2), end='')
        print('')
        total_train = 0
        optimizer.zero_grad()
        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the loss
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], data.size(0))
            top5.update(acc5[0], data.size(0))
            # record validation loss
            valid_losses.append(loss.item())

            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()  # get maximum probability of class
            total_val += len(data)

        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        valid_acc = correct / total_val * 100.

        epoch_len = len(str(epochs))

        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     # f'valid_loss: {valid_loss:.5f}, accuracy: {valid_acc:.4f}%')
                     f'valid_loss: {valid_loss:.5f}, accuracy: {correct}/{total_val} ({100. * correct / total_val:.3f}%)')

        print(print_msg)
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        correct = 0
        total_val = 0

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_acc, model)
        valid_accu.append(valid_acc)
        if early_stopping.early_stop:
            print("Early stopping")
            break    # load the last checkpoint with the best model
    model.load_state_dict(torch.load(path))

    return valid_accu
def test(model, device, test_loader, criterion):
    """
        Evaluates the performance of the trained model on a test dataset.

        Args:
            model: The neural network model to be evaluated.
            device: The device (CPU or GPU) on which to perform the evaluation.
            test_loader: DataLoader for the test dataset.
            criterion: The loss function used for evaluating the model's performance.
        Returns:
            test_loss: The average loss of the model on the test dataset.
            accuracy: The percentage of correct predictions over the test dataset.
        """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # test_loss += criterion(output, target).item()
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item() # get maximum probability of class

    test_loss /= len(test_loader.dataset)

    print('\nTest: average loss: {:.4f}, accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss, correct / len(test_loader.dataset)