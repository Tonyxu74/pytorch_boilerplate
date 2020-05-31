from utils.utils import *
from utils.dataset import generate_iterator
from utils.model import NewModel
from torch import optim, nn
from tqdm import tqdm


def train(continue_train=False):
    """
    Trains a PyTorch model
    :param continue_train: whether to continue training the model from a checkpoint defined by args.start_epoch
    :return: None
    """

    # the base name of the model being trained, useful for recording history
    base_model_name = args.model_name

    init_session_history(base_model_name)

    # create iterators
    train_iter = generate_iterator('./data', set_to_use='train')
    val_iter = generate_iterator('./data', set_to_use='val')

    # get model
    model = NewModel()

    # get optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(args.beta1, args.beta2)
    )
    lossfn = nn.CrossEntropyLoss()

    # if training model from previous saved weights
    if continue_train:
        model = load_weights(base_model_name, model, args.start_epoch)

    # move model to GPU
    if torch.cuda.is_available():
        model = model.cuda()
        lossfn = lossfn.cuda()

    # these values will be plotted versus epoch
    train_plot_loss = []
    val_plot_loss = []
    train_plot_acc = []
    val_plot_acc = []
    train_plot_f1 = []
    val_plot_f1 = []
    plot_epoch = []

    for epoch in range(args.start_epoch, args.num_epochs):

        # ==================== Training set ====================

        # progress bar to view progression of model
        train_pbar = tqdm(train_iter)

        # used to check accuracy to gauge model progression on training set
        train_losses_sum = 0
        train_n_total = 1
        train_pred_classes = []
        train_ground_truths = []

        for data, labels in train_pbar:

            # move to GPU
            if torch.cuda.is_available():
                data = data.cuda()
                labels = labels.cuda()

            # get predictions and loss for train
            pred_class, labels, loss = get_pred_and_loss(model, lossfn, data, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # predictions to check for model progression
            train_pred_classes.extend(pred_class)
            train_ground_truths.extend(labels)

            train_losses_sum += loss.item()

            # denotes average loss per item
            train_pbar.set_description(
                'Train || Epoch: {} || Loss: {:.5f} '.format(epoch, train_losses_sum / train_n_total)
            )
            train_n_total += 1

        # get metrics for train
        train_pred_classes = np.asarray(train_pred_classes)
        train_ground_truths = np.asarray(train_ground_truths)

        train_accuracy, train_conf_matrix, train_f1, train_precision, train_recall = get_metrics(
            train_pred_classes, train_ground_truths
        )

        # confusion matrix
        print(f'train confusion matrix: \n{train_conf_matrix}')

        # ==================== Validation set ====================

        # change modulo to do validation every few epochs
        if epoch % 1 == 0:

            # evaluate
            with torch.no_grad():
                model.eval()

                # progress bar to view progression of model
                val_pbar = tqdm(val_iter)

                # used to check accuracy to gauge model progression on validation set
                val_losses_sum = 0
                val_n_total = 1
                val_pred_classes = []
                val_ground_truths = []

                for data, labels in val_pbar:

                    # move to GPU
                    if torch.cuda.is_available():
                        data = data.cuda()
                        labels = labels.cuda()

                    # get predictions and loss for val
                    pred_class, labels, loss = get_pred_and_loss(model, lossfn, data, labels)

                    # predictions to check for model progression
                    val_pred_classes.extend(pred_class)
                    val_ground_truths.extend(labels)

                    val_losses_sum += loss.item()

                    # average val loss per item
                    val_pbar.set_description(
                        'Val || Epoch: {} || Loss: {:.5f} '.format(epoch, val_losses_sum / val_n_total)
                    )
                    val_n_total += 1

                # metrics for validation set
                val_pred_classes = np.asarray(val_pred_classes)
                val_ground_truths = np.asarray(val_ground_truths)
                val_accuracy, val_conf_matrix, val_f1, val_precision, val_recall = get_metrics(
                    val_pred_classes, val_ground_truths
                )

                # confusion matrix
                print(f'val confusion matrix: \n{val_conf_matrix}')

                model.train()

            print('Epoch: {} || Train_Acc: {} || Train_Loss: {} || Val_Acc: {} || Val_Loss: {}'.format(
                epoch, train_accuracy, train_losses_sum / train_n_total, val_accuracy, val_losses_sum / val_n_total
            ))

        train_plot_loss.append(train_losses_sum / train_n_total)
        val_plot_loss.append(val_losses_sum / val_n_total)
        train_plot_acc.append(train_accuracy)
        val_plot_acc.append(val_accuracy)
        train_plot_f1.append(train_f1)
        val_plot_f1.append(val_f1)
        plot_epoch.append(epoch)

        # change modulo number to save every few epochs
        if epoch % 1 == 0:
            # save weights
            model_name = save_weights(base_model_name, model, epoch, optimizer)

            # write history file
            write_history(
                model_name,
                train_losses_sum / train_n_total,
                val_losses_sum / val_n_total,
                train_accuracy,
                val_accuracy,
                train_f1,
                val_f1,
                train_precision,
                val_precision,
                train_recall,
                val_recall
            )

    plot_curves(
        base_model_name,
        train_plot_loss,
        val_plot_loss,
        train_plot_acc,
        val_plot_acc,
        train_plot_f1,
        val_plot_f1,
        plot_epoch
    )


