# based on hw2

import abc
import os
import sys
import tqdm
import torch
import numpy as np

from torch.utils.data import DataLoader
from typing import Callable, Any
from train_results import BatchResult, EpochResult, FitResult


class Trainer(abc.ABC):
    """
    A class abstracting the various tasks of training models.

    Provides methods at multiple levels of granularity:
    - Multiple epochs (fit)
    - Single epoch (train_epoch/test_epoch)
    - Single batch (train_batch/test_batch)
    """
    def __init__(self, model, loss_fn, optimizer, device=None):
        """
        Initialize the trainer.
        :param model: Instance of the model to train.
        :param loss_fn: The loss function to evaluate with.
        :param optimizer: The optimizer to train with.
        :param device: torch.device to run training on (CPU or GPU).
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

        if self.device:
            model.to(device=self.device)

    def fit(self, dataloaders, dl_test: DataLoader,
            num_epochs, checkpoints: str = None,
            early_stopping: int = None,
            print_every=1, **kw) -> FitResult:
        """
        Trains the model for multiple epochs with a given training set,
        and calculates validation loss over a given validation set.
        :param dl_train: Dataloader for the training set.
        :param dl_test: Dataloader for the test set.
        :param num_epochs: Number of epochs to train for.
        :param checkpoints: Whether to save model to file every time the
            test set accuracy improves. Should be a string containing a
            filename without extension.
        :param early_stopping: Whether to stop training early if there is no
            test loss improvement for this number of epochs.
        :param print_every: Print progress every this number of epochs.
        :return: A FitResult object containing train and test losses per epoch.
        """
        actual_num_epochs = 0
        train_loss, train_acc, test_loss, test_acc = [], [], [], []

        best_acc = None
        best_test_loss = np.inf
        epochs_without_improvement = 0
        dl_index = 0

        for epoch in range(num_epochs):
            verbose = False  # pass this to train/test_epoch.
            if epoch % print_every == 0 or epoch == num_epochs-1:
                verbose = True
            self._print(f'--- EPOCH {epoch+1}/{num_epochs} ---', verbose)

            # TODO: Train & evaluate for one epoch
            #  - Use the train/test_epoch methods.
            #  - Save losses and accuracies in the lists above.
            #  - Implement early stopping. This is a very useful and
            #    simple regularization technique that is highly recommended.
            #  - Optional: Implement checkpoints. You can use torch.save() to
            #    save the model to the file specified by the checkpoints
            #    argument.
            # ====== YOUR CODE: ======
            dl_index += 1
            if dl_index == len(dataloaders):
               dl_index = 0
            dl_train = dataloaders[dl_index]
            train_epoch_result = self.train_epoch(dl_train, verbose=verbose)
            avg_train_loss = sum(train_epoch_result.losses) / len(train_epoch_result.losses)
            #train_loss += train_epoch_result.losses
            train_loss.append(avg_train_loss)
            train_acc.append(train_epoch_result.accuracy)
            
            test_epoch_result = self.test_epoch(dl_test, verbose=verbose)
            avg_test_loss = sum(test_epoch_result.losses) / len(test_epoch_result.losses)
            if epoch > 0 and avg_test_loss >= best_test_loss:
                epochs_without_improvement += 1
            else:
                best_test_loss = avg_test_loss
                epochs_without_improvement = 0
                actual_num_epochs = epoch
                if checkpoints:
                    torch.save(self.model, checkpoints)
            test_loss.append(avg_test_loss)
            test_acc.append(test_epoch_result.accuracy)
            if early_stopping and epochs_without_improvement >= early_stopping:
               break
            # ========================

        return FitResult(actual_num_epochs,
                         train_loss, train_acc, test_loss, test_acc)

    def train_epoch(self, dl_train: DataLoader, **kw) -> EpochResult:
        """
        Train once over a training set (single epoch).
        :param dl_train: DataLoader for the training set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(True)  # set train mode
        self.model.train_mode = True
        return self._foreach_batch(dl_train, self.train_batch, **kw)

    def test_epoch(self, dl_test: DataLoader, **kw) -> EpochResult:
        """
        Evaluate model once over a test set (single epoch).
        :param dl_test: DataLoader for the test set.
        :param kw: Keyword args supported by _foreach_batch.
        :return: An EpochResult for the epoch.
        """
        self.model.train(False)  # set evaluation (test) mode
        return self._foreach_batch(dl_test, self.test_batch, **kw)

    @abc.abstractmethod
    def train_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model, calculates loss,
        preforms back-propagation and uses the optimizer to update weights.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def test_batch(self, batch) -> BatchResult:
        """
        Runs a single batch forward through the model and calculates loss.
        :param batch: A single batch of data  from a data loader (might
            be a tuple of data and labels or anything else depending on
            the underlying dataset.
        :return: A BatchResult containing the value of the loss function and
            the number of correctly classified samples in the batch.
        """
        raise NotImplementedError()

    @staticmethod
    def _print(message, verbose=True):
        """ Simple wrapper around print to make it conditional """
        if verbose:
            print(message)

    @staticmethod
    def _foreach_batch(dl: DataLoader,
                       forward_fn: Callable[[Any], BatchResult],
                       verbose=True, max_batches=None) -> EpochResult:
        """
        Evaluates the given forward-function on batches from the given
        dataloader, and prints progress along the way.
        """
        losses = []
        af_losses = []
        nsr_losses = []
        num_correct = 0
        num_samples = len(dl.sampler)
        num_batches = len(dl.batch_sampler)

        if max_batches is not None:
            if max_batches < num_batches:
                num_batches = max_batches
                num_samples = num_batches * dl.batch_size

        if verbose:
            pbar_file = sys.stdout
        else:
            pbar_file = open(os.devnull, 'w')

        pbar_name = forward_fn.__name__
        with tqdm.tqdm(desc=pbar_name, total=num_batches,
                       file=pbar_file) as pbar:
            dl_iter = iter(dl)
            for batch_idx in range(num_batches):
                data = next(dl_iter)
                batch_res = forward_fn(data)

                pbar.set_description(f'{pbar_name} ({batch_res.loss:.3f})')
                pbar.update()

                losses.append(batch_res.loss)
                af_losses.append(batch_res.af_loss)
                nsr_losses.append(batch_res.nsr_loss)
                num_correct += batch_res.num_correct

            avg_loss = sum(losses) / num_batches
            accuracy = 100. * num_correct / num_samples
            n_segments = batch_res.n_segments
            n_true_positive = batch_res.n_true_positive
            n_true_negative = batch_res.n_true_negative
            pbar.set_description(f'{pbar_name} '
                                 f'(Avg. Loss {avg_loss:.3f}, '
                                 f'#Segments {n_segments:.1f}, '
                                 f'True positive {n_true_positive:.1f}, '
                                 f'True negative {n_true_negative:.1f}), '
                                 )

        return EpochResult(losses=losses, accuracy=accuracy, af_losses=af_losses, nsr_losses=nsr_losses)

    def zero_counters(self):
      self.n_true_positive = 0
      self.n_true_negative = 0
      self.n_false_positive = 0
      self.n_false_negative = 0
      self.num_AF = 0
      self.num_NSR = 0
      self.n_segments = 0

    def print_stats(self, phaze):
      print('Number of AF', self.num_AF)
      print('Number of NSR', self.num_NSR)
      print('True af:', self.n_true_positive, f'{100*self.n_true_positive/self.n_segments:.1f}%')
      print('True nsr:', self.n_true_negative, f'{100*self.n_true_negative/self.n_segments:.1f}%')
      print('False af:', self.n_false_positive, f'{100*self.n_false_positive/self.n_segments:.1f}%')
      print('False nsr:', self.n_false_negative, f'{100*self.n_false_negative/self.n_segments:.1f}%')
      print('Number of segments', self.n_segments)
      print(f'{phaze} accuracy: {100*(self.n_true_positive+self.n_true_negative)/self.n_segments:.1f}%')
      n = self.n_true_positive + self.n_false_negative
      if n > 0:
        print(f'{phaze}, True Positive Rate (recall): {100*self.n_true_positive/n:.1f}%');



class TorchTrainer(Trainer):
    def __init__(self, model, loss_fn, optimizer, device=None):
        super().__init__(model, loss_fn, optimizer, device)
        self.num_AF = 0
        self.num_NSR = 0
        self.n_segments = 0
        self.n_true_positive = 0
        self.n_true_negative = 0
        self.n_false_positive = 0
        self.n_false_negative = 0

    def train_batch(self, batch) -> BatchResult:
        X, y = batch

        self.optimizer.zero_grad()

        # Forward pass
        y_pred_scores = self.model(X)

        #print(y_pred_scores.shape) # (N, 2)
        #print(y_pred_scores.shape) # (N, seq_len, 2) when not using attention

        # Backward pass
        #loss = self.loss_fn(y_pred_scores, y)

        #y_pred_scores = y_pred_scores.view(-1, 2)
        #y = y.view(-1)
        loss = self.loss_fn(y_pred_scores, y)

        #a = list(self.model.parameters()) # for equal tests
        #a = [x.clone() for x in a]

        loss.backward()

        # Weight updates
        self.optimizer.step()

        #b = list(self.model.parameters()) # for equal tests
        #b = [x.clone() for x in b]
        #for i in range(len(a)):
           #print('equal test:', torch.equal(a[i].data, b[i].data))
       
        y_pred = torch.argmax(y_pred_scores, dim=1)
        num_correct = torch.sum(y_pred == y).float().item()
        loss = loss.item()
        af_loss = 0
        nsr_loss = 0

        #print(X.shape) # (N, seq_len, H, W)
        #print(y.shape) # (N)
        #print(y_pred_scores.shape) # (N, 2)

        num_AF = torch.sum(y == 1).item()
        num_NSR = torch.sum(y == 0).item()
        self.num_AF += num_AF
        self.num_NSR += num_NSR

        n_segments = len(y)
        n_true_positive = torch.sum(torch.min((y == 1), (y_pred == 1))).item()
        n_true_negative = torch.sum(torch.min((y == 0), (y_pred == 0))).item()
        n_false_positive = torch.sum(torch.min((y == 0), (y_pred == 1))).item()
        n_false_negative = torch.sum(torch.min((y == 1), (y_pred == 0))).item()

        self.n_segments += n_segments
        self.n_true_positive += n_true_positive
        self.n_true_negative += n_true_negative
        self.n_false_positive += n_false_positive
        self.n_false_negative += n_false_negative

        result = BatchResult(loss, num_correct, n_segments, n_true_positive, n_true_negative, n_false_positive, n_false_negative, af_loss, nsr_loss)
        return result

    def test_batch(self, batch) -> BatchResult:
        X, y = batch

        with torch.no_grad():
            y_pred_scores = self.model(X)
            loss = self.loss_fn(y_pred_scores, y)
            y_pred = torch.argmax(y_pred_scores, dim=1)
            num_correct = torch.sum(y_pred == y).float().item()
            loss = loss.item()
            af_loss = 0
            nsr_loss = 0
            num_AF = torch.sum(y == 1).item()
            num_NSR = torch.sum(y == 0).item()
            self.num_AF += num_AF
            self.num_NSR += num_NSR
            n_segments = len(y)
            n_true_positive = torch.sum(torch.min((y == 1), (y_pred == 1))).item()
            n_true_negative = torch.sum(torch.min((y == 0), (y_pred == 0))).item()
            n_false_positive = torch.sum(torch.min((y == 0), (y_pred == 1))).item()
            n_false_negative = torch.sum(torch.min((y == 1), (y_pred == 0))).item()

            self.n_segments += n_segments
            self.n_true_positive += n_true_positive
            self.n_true_negative += n_true_negative
            self.n_false_positive += n_false_positive
            self.n_false_negative += n_false_negative

            # ========================

        result = BatchResult(loss, num_correct, n_segments, n_true_positive, n_true_negative, n_false_positive, n_false_negative, af_loss, nsr_loss)
        return result
