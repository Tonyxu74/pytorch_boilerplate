import argparse

parser = argparse.ArgumentParser()

# ======= Model parameters =======

parser.add_argument('--model_name', default='resnet18', type=str,
                    help='pretrained model name')

# ======= Training parameters =======

parser.add_argument('--lr', default=0.001, type=float,
                    help='learning rate')
parser.add_argument('--weight_decay', default=0.0001, type=float,
                    help='weight decay/weights regularizer for sgd')
parser.add_argument('--beta1', default=0.9, type=float,
                    help='momentum for sgd, beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float,
                    help='momentum for sgd, beta1 for adam')

parser.add_argument('--num_epochs', default=10, type=int,
                    help='epochs to train for')
parser.add_argument('--start_epoch', default=3, type=int,
                    help='epoch to start training. useful if continue from a checkpoint')
parser.add_argument('--eval_epoch', default=3, type=int,
                    help='epoch to start evaluating. useful if continue from a checkpoint')

# ======= Dataloader parameters =======

parser.add_argument('--batch_size', default=128, type=int,
                    help='input batch size')
parser.add_argument('--workers', default=8, type=int,
                    help='number of workers to use for GenerateIterator')


args = parser.parse_args()