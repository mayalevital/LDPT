import numpy as np
import random
import wfdb
import torch
from torch.utils.data import Dataset
import torch.optim as optim
import torch.nn as nn
import copy
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
import os

# our libs:
import datasets
import main_sig_process
import main_sig_process_rr
import main_paper_net
from training import TorchTrainer
from train_results import BatchResult, EpochResult, FitResult

def train_epoch_with_trainer(model, trainer, base_dir, patient_list, device, rr=False):
  batch_size = 4

  total_loss = 0
  i_patient = 1
  model.train()
  for patient in patient_list:
    folder = 'AF_data/' + patient
    print('training patient #', i_patient, ', folder:', folder)
    if rr:
      dataset = datasets.SegmentDataset(base_dir + patient, device=device, rr=True)
    else:
      dataset = datasets.SegmentDataset(base_dir + patient, device=device)
    dl_train = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    train_epoch_result = trainer.train_epoch(dl_train, verbose=True)
    total_loss += sum(train_epoch_result.losses)
    i_patient += 1
  return total_loss

def test_epoch_with_trainer(model, trainer, base_dir, patient_list, device, rr=False):
  batch_size = 4

  total_loss = 0
  i_patient = 1
  model.eval()
  for patient in patient_list:
    folder = 'AF_data/' + patient
    print('testing patient #', i_patient, ', folder:', folder)
    if rr:
      dataset = datasets.SegmentDataset(base_dir + patient, device=device, rr=True)
    else:
      dataset = datasets.SegmentDataset(base_dir + patient, device=device)
    dl_test = torch.utils.data.DataLoader(dataset, batch_size, shuffle=True)
    test_epoch_result = trainer.test_epoch(dl_test, verbose=True)
    total_loss += sum(test_epoch_result.losses)
    i_patient += 1
  return total_loss

def train_with_trainer(model, base_dir='wavelets/', patient_list=None, test_list=None, num_epochs=100, device='cuda', optimizer=None, rr=False, lr=1e-4, previous_result=None):
  if not optimizer:
     optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
  #weight = torch.tensor([1.0, 25.0], device=device)
  #loss_fn = nn.CrossEntropyLoss(weight=weight)
  loss_fn = nn.CrossEntropyLoss()
  trainer = TorchTrainer(model, loss_fn, optimizer, device=device)
  start_epoch = model.epochs_performed
  train_loss = []
  test_loss = []
  train_acc = []
  test_acc = []

  for epoch in range(start_epoch, start_epoch+num_epochs):
    trainer.zero_counters()
    total_train_losss = train_epoch_with_trainer(model, trainer, base_dir, patient_list, device, rr=rr)
    train_loss.append(total_train_losss)
    train_acc.append((trainer.n_true_positive+trainer.n_true_negative)/trainer.n_segments)
    trainer.print_stats('Train');
    print(f'total train loss: {total_train_losss:.3f}')
    print('end of train epoch #', epoch)

    trainer.zero_counters()
    total_test_losss = test_epoch_with_trainer(model, trainer, base_dir, test_list, device, rr=rr)
    test_loss.append(total_test_losss)
    test_acc.append((trainer.n_true_positive+trainer.n_true_negative)/trainer.n_segments)
    trainer.print_stats('Test');
    print(f'total test loss: {total_test_losss:.3f}')

    model.epochs_performed += 1
    actual_num_epochs = epoch
    print('end of test epoch #', epoch)

  if previous_result:
     train_loss = previous_result.train_loss + train_loss
     train_acc = previous_result.train_acc + train_acc
     test_loss = previous_result.test_loss + test_loss
     test_acc = previous_result.test_acc + test_acc
  result = FitResult(actual_num_epochs, train_loss, train_acc, test_loss, test_acc)

  return model, result

def train_interval_net(model, base_dir='wavelets/', patient_list=None, test_list=None, num_epochs=100, device='cuda', optimizer=None, rr=True, lr=1e-5):
  return train_with_trainer(model, base_dir=base_dir, patient_list=patient_list, test_list=test_list, num_epochs=num_epochs, device=device, optimizer=optimizer, rr=rr, lr=lr)

# orig_model should be of class MainNet
def transfer_learning(orig_model, patient_list=None, test_list=None, base_dir='wavelets/', num_epochs=100, device='cuda'):
  model = main_paper_net.IntervalNet(last_layers=copy.deepcopy(orig_model.last_layers))
  #params = model.parameters()
  params = [
    dict(params=model.lstm.parameters(), lr=1e-5),
    dict(params=model.fc.parameters(), lr=1e-5),
    #dict(params=model.last_layers.parameters(), lr=1e-6),
    dict(params=model.last_layers.rnn.parameters(), lr=0),
    dict(params=model.last_layers.fc2.parameters(), lr=0), # attention
    dict(params=model.last_layers.fc3.parameters(), lr=1e-5),
  ]
  optimizer = optim.Adam(params, weight_decay=1e-3)
  
  return train_with_trainer(model, patient_list=patient_list, test_list=test_list, base_dir=base_dir, num_epochs=num_epochs, device=device, optimizer=optimizer, rr=True)

def signal_processsing(folder):
  extension = 'atr'
  signal, fields = wfdb.rdsamp(folder)
  #signal, fields = wfdb.rdsamp(folder, sampto=10*60*250*4) # only read part of the signal, for code tests
  #print('signal.shape:', signal.shape) # (9205760, 2) for AF_data/04746 => 36823.04 seconds => 614 minutes => 10.2 hours | 614 min => 61 segments
  ann = wfdb.rdann(folder, extension)
  fs = fields['fs']
       
  sig = main_sig_process.sig_process(signal, ann, fs)
  return sig

def signal_processsing_rr(folder):
  extension = 'atr'
  signal, fields = wfdb.rdsamp(folder)
  ann = wfdb.rdann(folder, extension)
  fs = fields['fs']
       
  sig = main_sig_process_rr.sig_process(signal, ann, fs)
  return sig

def samples_in_record(record_name):
  record = wfdb.rdheader(record_name)
  return record.sig_len

def plot(folder, sampfrom, sampto):
  record = wfdb.rdrecord(folder, sampfrom=sampfrom, sampto=sampto)
  ann = wfdb.rdann(folder, 'atr', sampfrom=sampfrom, sampto=sampto)
  sample = ann.sample - sampfrom
  wfdb.plot_items(signal=record.p_signal,
                    ann_samp=[sample, sample],
                    ann_sym=[ann.aux_note, ann.aux_note],
                    fs=record.fs,
                    title='test plot', time_units='seconds',
                    figsize=(20,8))
  print('record.fs', record.fs)
  print('ann.sample:', ann.sample)
  print('ann.aux_note:', ann.aux_note)

def find_fs():
  list = wfdb.get_record_list('afdb') # all fs are 250
  list = wfdb.get_record_list('nsrdb') # all fs are 128
  #print(list)
  list.remove('00735')
  list.remove('03665')
  for patient in list:
    #folder = 'AF_data/' + patient
    folder = 'NSR_data/' + patient
    sig_t, fields = wfdb.rdsamp(folder, sampto=1)
    print(patient, ': ', fields["fs"])

def get_patient_list():
  return ['04048', '04908', '04936', '06426', '07162', '07859', '08378', '05121', '04043']

def get_test_list():
  return ['08405', '08455']

def test_amount():
  ann = wfdb.rdann('AF_data/04043', 'atr')
  x1 = 500
  x2 = 380000
  amount = main_sig_process.sig_process.non_normal_amount(ann, 0, 500)
  assert(amount == 0)
  amount = main_sig_process.sig_process.non_normal_amount(ann, 500, 270000)
  assert(amount == 270000-266498)
  amount = main_sig_process.sig_process.non_normal_amount(ann, 500, 380000)
  assert(amount == 376328 - 266498)
  amount = main_sig_process.sig_process.non_normal_amount(ann, 270000, 380000)
  assert(amount == 376328 - 270000)
  amount = main_sig_process.sig_process.non_normal_amount(ann, 2500000, 2800000)
  assert(amount == 2602516-2585284 + 2739812-2634911 + 2779581-2745162)
  #print('amount:', amount)
  print('PASS')

def plot_fit(fit_res: FitResult, fig=None, log_loss=False, legend=None):
    """
    Plots a FitResult object.
    Creates four plots: train loss, test loss, train acc, test acc.
    :param fit_res: The fit result to plot.
    :param fig: A figure previously returned from this function. If not None,
        plots will the added to this figure.
    :param log_loss: Whether to plot the losses in log scale.
    :param legend: What to call this FitResult in the legend.
    :return: The figure.
    """
    if fig is None:
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 10),
                                 sharex='col', sharey=False)
        axes = axes.reshape(-1)
    else:
        axes = fig.axes

    for ax in axes:
        for line in ax.lines:
            if line.get_label() == legend:
                line.remove()

    p = itertools.product(['train', 'test'], ['loss', 'acc'])
    for idx, (traintest, lossacc) in enumerate(p):
        ax = axes[idx]
        attr = f'{traintest}_{lossacc}'
        data = getattr(fit_res, attr)
        h = ax.plot(np.arange(1, len(data) + 1), data, label=legend)
        ax.set_title(attr)
        if lossacc == 'loss':
            ax.set_xlabel('Iteration #')
            ax.set_ylabel('Loss')
            if log_loss:
                ax.set_yscale('log')
                ax.set_ylabel('Loss (log)')
        else:
            ax.set_xlabel('Epoch #')
            ax.set_ylabel('Accuracy (%)')
        if legend:
            ax.legend()

    return fig, axes

def compute_auc(model, test_list, base_dir = 'wavelets/', rr=False):
  probs = []
  ys = []

  with torch.no_grad():
    for patient in test_list:
      dataset = datasets.SegmentDataset(base_dir + patient, rr=rr)
      for X, y in dataset:
        X = X.unsqueeze(0)
        #print(X.shape) # [1, 20, 20, 300]
        p = model.probs(X)
        #print(p.shape) #[1, 100]
        p = p.squeeze(0)
        probs.append(p)
        ys.append(y)

  probs = torch.stack(probs)
  y = torch.stack(ys)
  probs = probs.cpu()
  y = y.cpu()

  #print(y.shape)
  #print(probs.shape)

  # keep probabilities for the positive outcome only
  probs = probs[:, 1] # roc_auc_score takes the scores of the last label for the binary case

  auc = roc_auc_score(y, probs)

  fpr, tpr, _ = roc_curve(y, probs)
  # plot the roc curve for the model
  plt.plot(fpr, tpr, marker='.', label='AUC')
  # axis labels
  plt.xlabel('False Positive Rate')
  plt.ylabel('True Positive Rate')
  # show the legend
  plt.legend()
  # show the plot
  plt.show()

  return auc

def calc_noisy():
  # calc percent of data removed because it was too noisy
  patient_list = ['04048', '04908', '04936', '06426', '07162', '07859', '08378', '05121', '04043', '08405', '08455']
  good_data_seconds = 0
  total_samples = 0
  for patient in patient_list:
    folder = 'AF_data/' + patient
    labels = torch.load('wavelets/' + patient + '/labels.dat')
    n_labels = len(labels)
    good_data_seconds += n_labels*10*60 # each label is 10 minutes of good data
    total_samples += samples_in_record(folder)
    print('patient:', patient, ', n_labels: ', n_labels)
  fs = 250
  all_data_seconds = total_samples / fs
  print('good_data_seconds:', good_data_seconds, ', all_data_seconds: ', all_data_seconds)
  print('%good data:', 100*good_data_seconds/all_data_seconds)

def calc_measures(TP, TN, FP, FN):
  recall = TP / (TP+FN)
  precision = TP / (TP+FP)
  print('recall:', 100*recall, '%')
  print('precision:', 100*precision, '%')

def print_measures():
  print('wavelet train:')
  calc_measures(262, 177, 0, 0)
  print('wavelet test:')
  calc_measures(78, 27, 7, 0)
  print('interval train:')
  calc_measures(445, 243, 11, 2)
  print('interval test:')
  calc_measures(143, 44, 1, 0)
  print('transfer train:')
  calc_measures(440, 244, 10, 7)
  print('transfer test:')
  calc_measures(143, 44, 1, 0)

  print('paper 1 train:')
  n_segments = 291240
  n_af = 21542
  n_neg = n_segments - n_af
  spc = 0.95 # spc=TN/(TN+FP) => TN = spc*(TN+FP)
  TN = spc*n_neg
  FP = n_neg-TN
  acc = 0.94 # acc=(TP+TN)/N => TP+TN=acc*N => TP=acc*N-TN
  TP = acc*n_segments-TN
  FN = n_af-TP
  calc_measures(TP, TN, FP, FN)

  print('paper 1 test:')
  n_segments = 72772
  n_af = 5419
  n_neg = n_segments - n_af
  spc = 0.96
  TN = spc*n_neg
  FP = n_neg-TN
  acc = 0.96
  TP = acc*n_segments-TN
  FN = n_af-TP
  calc_measures(TP, TN, FP, FN)

  print('paper 2 train:')
  calc_measures(430615, 523241, 7040, 7407)
  print('paper 2 test:')
  calc_measures(91888, 65699, 255, 116)


def download_database():
  db_dir='afdb'
  dl_dir='AF_data'
  wfdb.io.dl_database(db_dir, dl_dir, records='all', annotators='all', keep_subdirs=True, overwrite=False)

def preprocess():
  print('Preprocessing')
  patient_list = get_patient_list()
  test_list = get_test_list()
  patient_list = patient_list + test_list
  for patient in patient_list:
    print('patient:', patient)
    folder = 'AF_data/' + patient
    if not os.path.isdir('wavelets/' + patient):
      os.makedirs('wavelets/' + patient)
    if not os.path.isfile('wavelets/' + patient + '/wavelets.dat'):
      sig = signal_processsing(folder)
      sig.calc_and_save_wavelets('wavelets/' + patient)
    if not os.path.isfile('wavelets/' + patient + '/rr_intervals.dat'):
      sig = signal_processsing_rr(folder)
      sig.calc_and_save_wavelets('wavelets/' + patient)
  print('Preprocessing done')

def train():
  patient_list = get_patient_list()
  test_list = get_test_list()
  # wavelet model:
  model = main_paper_net.MainNet()
  model, result = train_with_trainer(model, patient_list=patient_list, test_list=test_list, num_epochs=150, previous_result=None)
  # RR-interval model:
  i_model = main_paper_net.IntervalNet()
  i_model, i_result = train_interval_net(i_model, patient_list=patient_list, test_list=test_list, num_epochs=150)
  # transfer learning:
  t_model, t_result = transfer_learning(model, patient_list=patient_list, test_list=test_list, num_epochs=150)
  torch.save(result, 'result150wavelets.dat')
  torch.save(i_result, 'result150intervals.dat')
  torch.save(t_result, 'result150transfer.dat')
  torch.save(model, 'model150wavelets.dat')
  torch.save(i_model, 'model150intervals.dat')
  torch.save(t_model, 'model150transfer.dat')

def show_results():
  model = torch.load('model150wavelets.dat')
  i_model = torch.load('model150intervals.dat')
  t_model = torch.load('model150transfer.dat')
  result = torch.load('result150wavelets.dat')
  i_result = torch.load('result150intervals.dat')
  t_result = torch.load('result150transfer.dat')
  plot_fit(result)
  plot_fit(i_result)
  plot_fit(t_result)

if __name__ == '__main__':
  download_database()
  preprocess()
  train()
  show_results()