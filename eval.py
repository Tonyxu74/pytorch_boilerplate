from utils.utils import *
from utils.dataset import generate_iterator
from utils.model import NewModel
from torch import nn
from tqdm import tqdm


def evaluate():
    """
    Evaluates on a PyTorch model
    NOTE: assumes that test labels are available, will require slight adjustment if not
    :return: None
    """
    base_model_name = args.model_name

    # create iterator
    test_iter = generate_iterator('./data', set_to_use='test')

    # get model and load weights
    model = NewModel()
    model = load_weights(base_model_name, model, args.eval_epoch)

    # lossfn
    lossfn = nn.CrossEntropyLoss()

    # move to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        lossfn = lossfn.cuda()

    # get progress bar
    test_pbar = tqdm(test_iter)

    # get test predictions
    test_pred_classes = []
    test_ground_truths = []
    test_losses_sum = 0
    test_n_total = 1

    with torch.no_grad():
        model.eval()

        for data, labels in test_pbar:

            # move to GPU
            if torch.cuda.is_available():
                data = data.cuda()
                labels = labels.cuda()

            # get predictions and loss for test
            pred_class, labels, loss = get_pred_and_loss(model, lossfn, data, labels)

            # predictions to check for model progression
            test_pred_classes.extend(pred_class)
            test_ground_truths.extend(labels)

            test_losses_sum += loss.item()

            # average test loss per item
            test_pbar.set_description(
                'Test || Loss: {:.5f} '.format(test_losses_sum / test_n_total)
            )
            test_n_total += 1

    # metrics for test set
    test_pred_classes = np.asarray(test_pred_classes)
    test_ground_truths = np.asarray(test_ground_truths)
    test_accuracy, test_conf_matrix, test_f1, test_precision, test_recall = get_metrics(
        test_pred_classes, test_ground_truths
    )

    # print results
    print('Test set || acc: {} || average loss: {} || f1 score: {} || precision: {} || recall: {}'.format(
        test_accuracy,
        test_losses_sum / test_n_total,
        test_f1,
        test_precision,
        test_recall
    ))

    # print confusion matrix
    print(f'test confusion matrix: \n{test_conf_matrix}')
