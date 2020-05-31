import os
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from myargs import args
import torch


def find_file(root_dir, contains):
    """
    Finds file with given root directory containing keyword "contains"
    :param root_dir: root directory to search in
    :param contains: the keyword that should be contained
    :return: a list of the file paths of found files
    """

    all_files = []
    for path, subdirs, files in os.walk(root_dir):
        for file in files:
            if contains in file:
                all_files.append(os.path.join(path, file))

    return all_files


def split_trainval(datapath, train_split=0.7):
    """
    Given a datapath, create a train/val split; useful if training and validation data are not split already
    :param datapath: path to data
    :param train_split: amount of data to allocate to train
    :return: a dict with lists of the train and validation paths
    """

    list_datapaths = find_file(datapath, '.file_extension')
    train_num = int(len(list_datapaths) * train_split)

    random.shuffle(list_datapaths)
    trainpaths = list_datapaths[:train_num]
    valpaths = list_datapaths[train_num:]

    datapaths = {'train': trainpaths, 'val': valpaths}

    return datapaths


def get_mean_std(path):
    """
    Given a path to a dataset, get the mean and standard deviation of the image data
    :param path: path to dataset
    :return: a mean and standard deviation computed for the whole dataset
    """

    list_datapaths = find_file(path, '.file_extension')

    values = []
    for path in list_datapaths:
        # open an image
        img = Image.open(path)
        np.asarray(img)

        values.append(img)

    values = np.asarray(values)

    # return a mean and standard deviation for each RGB(A) value
    return [np.mean(values[..., i]) for i in range(values.size[-1])], \
           [np.std(values[..., i]) for i in range(values.size[-1])]


def plot_curves(base_model_name, train_loss, val_loss, train_acc, val_acc, train_f1, val_f1, epochs):
    """
    Given progression of train/val loss/acc, plots curves
    :param base_model_name: name of base model in training session
    :param train_loss: the progression of training loss
    :param val_loss: the progression of validation loss
    :param train_acc: the progression of training accuracy
    :param val_acc: the progression of validation accuracy
    :param train_f1: the progression of training f1 score
    :param val_f1: the progression of validation f1 score
    :param epochs: epochs that model ran through
    :return: None
    """

    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.plot(epochs, train_loss, label='train loss')
    plt.plot(epochs, val_loss, label='val loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.title('Loss curves')
    plt.legend()

    plt.subplot(132)
    plt.plot(epochs, train_acc, label='train accuracy')
    plt.plot(epochs, val_acc, label='val accuracy')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title('Accuracy curves')
    plt.legend()

    plt.subplot(133)
    plt.plot(epochs, train_f1, label='train f1 score')
    plt.plot(epochs, val_f1, label='val f1 score')
    plt.xlabel('epochs')
    plt.ylabel('f1 score')
    plt.title('f1 curves')
    plt.legend()

    plt.suptitle(f'Session: {base_model_name}')

    plt.savefig('previous_run.png')
    plt.show()


def get_metrics(pred_classes, ground_truths):
    """
    Given two linear arrays of predicted classes and ground truths, return accuracy, confusion matrix, f1 score,
    precision and recall
    :param pred_classes: classes predicted by model
    :param ground_truths: ground truths for predictions
    :return: tuple of accuracy, confusion matrix, f1, precision, recall
    """

    accuracy = np.mean((pred_classes == ground_truths)).astype(np.float)
    cm = confusion_matrix(ground_truths, pred_classes, labels=[0, 1])
    f1 = f1_score(ground_truths, pred_classes, labels=[0, 1])
    precision = precision_score(ground_truths, pred_classes, labels=[0, 1])
    recall = recall_score(ground_truths, pred_classes, labels=[0, 1])

    return accuracy, cm, f1, precision, recall


def get_pred_and_loss(model, lossfunc, model_input, labels):
    """
    given model and loss function, calculates predictions and loss for input and ground truth, returns list of
    predictions and labels
    :param model: model to get predictions from
    :param lossfunc: loss function to calculate loss
    :param model_input: input data for model
    :param labels: ground truth labels for input
    :return: a tuple of the predicted classes in a list, labels in a list, and loss
    """

    # get prediction
    prediction = model(model_input)

    # get loss
    loss = lossfunc(prediction, labels)

    pred_class = torch.argmax(prediction.detach(), dim=-1)
    pred_class = pred_class.cpu().numpy().tolist()
    labels = labels.detach().cpu().numpy().tolist()

    return pred_class, labels, loss


def init_session_history(base_model_name):
    """
    Initializes a section in the history file for current training session
    :param base_model_name: the model base name
    :return: None
    """

    with open('history.txt', 'a+') as hist_fp:
        hist_fp.write(
            '\n============================== Base_model: {} ==============================\n'.format(base_model_name)

            + 'arguments: {}\n'.format(args)
        )


def write_history(
        model_name,
        train_loss,
        val_loss,
        train_acc,
        val_acc,
        train_f1,
        val_f1,
        train_precision,
        val_precision,
        train_recall,
        val_recall
):
    """
    Write a history.txt file for each model checkpoint
    :param model_name: name of the current model checkpoint
    :param train_loss: the training loss for current checkpoint
    :param val_loss: the validation loss for current checkpoint
    :param train_acc: the training accuracy for current checkpoint
    :param val_acc: the validation accuracy for current checkpoint
    :param train_f1: the training f1 score for current checkpoint
    :param val_f1: the validation f1 score for current checkpoint
    :param train_precision: the training precision score for current checkpoint
    :param val_precision: the validation precision score for current checkpoint
    :param train_recall: the training recall score for current checkpoint
    :param val_recall: the validation recall score for current checkpoint
    :return: None
    """

    with open('history.txt', 'a') as hist_fp:
        hist_fp.write(
            '\ncheckpoint name: {} \n'.format(model_name)

            + 'train loss: {} || train accuracy: {} || train f1: {} || train precision: {} || train recall: {}\n'.format(
                round(train_loss, 5),
                round(train_acc, 5),
                round(train_f1, 5),
                round(train_precision, 5),
                round(train_recall, 5)
            )

            + 'val loss: {} || val accuracy: {} || val f1: {} || val precision: {} || val recall: {}\n'.format(
                round(val_loss, 5),
                round(val_acc, 5),
                round(val_f1, 5),
                round(val_precision, 5),
                round(val_recall, 5)
            )
        )


def save_weights(base_model_name, model, epoch, optimizer):
    """
    Saves a state dictionary given a model, the epoch it has trained to, and the optimizer
    :param base_model_name: name of base model in training session
    :param model: weights will be saved for this model
    :param epoch: epoch model has trained to
    :param optimizer: optimizer used during training
    :return: Name of the model that was saved, useful for writing history file
    """
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    if not os.path.exists('./prev_models/'):
        os.mkdir('./prev_models/')

    model_name = '{}_{}'.format(base_model_name, epoch)
    torch.save(state, './prev_models/{}.pt'.format(model_name))
    return model_name


def load_weights(base_model_name, model, epoch):
    """
    Loads previously trained weights into a model given an epoch and the model itself
    :param base_model_name: name of base model in training session
    :param model: the model to load weights into
    :param epoch: what epoch of training to load
    :return: the model with weights loaded in
    """

    pretrained_dict = torch.load('./prev_models/{}_{}.pt'.format(base_model_name, epoch))['state_dict']
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    return model


def read_history():
    """
    Reads history file and prints out plots for each training session
    :return: None
    """

    with open('history.txt', 'r') as hist:

        # get all lines
        all_lines = hist.readlines()

        # remove newlines for easier processing
        rem_newline = []
        for line in all_lines:
            if len(line) == 1 and line == '\n':
                continue
            rem_newline.append(line)

        # get individual training sessions
        base_names = []
        base_indices = []
        for i in range(len(rem_newline)):
            if rem_newline[i][0] == '=':
                base_names.append(rem_newline[i].replace('=', '').split(' ')[-2])
                base_indices.append(i)

        # create plots for each individual session
        for i in range(len(base_names)):
            name = base_names[i]

            # get last session
            if i == len(base_names) - 1:
                session_data = rem_newline[base_indices[i]:]

            # get session
            else:
                session_data = rem_newline[base_indices[i]: base_indices[i + 1]]

            # now generate the plots
            train_plot_loss = []
            val_plot_loss = []
            train_plot_acc = []
            val_plot_acc = []
            train_plot_f1 = []
            val_plot_f1 = []
            plot_epoch = []

            for line in session_data:

                # case for getting checkpoint epoch
                if 'checkpoint' in line:
                    plot_epoch.append(int(line.split('_')[-2]))

                # case for getting train data for epoch
                elif 'train' in line:
                    train_plot_loss.append(float(line.split(' ')[2]))
                    train_plot_acc.append(float(line.split(' ')[6]))
                    train_plot_f1.append(float(line.split(' ')[10]))

                # case for getting val data for epoch
                elif 'val' in line:
                    val_plot_loss.append(float(line.split(' ')[2]))
                    val_plot_acc.append(float(line.split(' ')[6]))
                    val_plot_f1.append(float(line.split(' ')[10]))

            # plot
            plot_curves(
                name,
                train_plot_loss,
                val_plot_loss,
                train_plot_acc,
                val_plot_acc,
                train_plot_f1,
                val_plot_f1,
                plot_epoch
            )