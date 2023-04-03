from dataset import *
import argparse
from models import *
import yaml
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import *
import warnings
from transformers import AutoModelForAudioClassification
warnings.filterwarnings('ignore', message='No positive class found in y_true') # positive class is rare in y_true
# python3 /Users/yuxiaoliu/miniconda3/envs/si699-music-tagging/lib/python3.10/site-packages/tensorboard/main.py --logdir=runs
import matplotlib.pyplot as plt
import logging
logging.basicConfig(filename="log",
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    filemode='w',
                    level=logging.INFO)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--tag_file', type=str, default='data/autotagging_top50tags.tsv')
parser.add_argument('--npy_root', type=str, default='data/npy')
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--num_epochs', type=int, default=5)
parser.add_argument('--model', type=str, default='samplecnn')
parser.add_argument('--threshold', type=float, default=0.5)
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Run on:", device)


def train(model, epoch, criterion, optimizer, train_loader):
    losses = []
    ground_truth = []
    prediction = []
    model.train()
    for input, label in tqdm(train_loader):
        input, label = input.to(device), label.to(device)
        # input = input.unsqueeze(1)
        output = model(input)
        # print("label:", label.shape, "output", output.shape)
        loss = criterion(output, label.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        ground_truth.append(label)
        prediction.append(output)
    get_eval_metrics(prediction, ground_truth, 'train', epoch, losses)
    return model


@torch.no_grad()
def validate(model, epoch, criterion, val_loader):
    losses = []
    ground_truth = []
    prediction = []
    model.eval()
    for input, label in tqdm(val_loader):
        input, label = input.to(device), label.to(device)
        # input = input.unsqueeze(1)
        output = model(input)
        loss = criterion(output, label.float())
        losses.append(loss.detach())
        ground_truth.append(label)
        prediction.append(output)
    get_eval_metrics(prediction, ground_truth, 'val', epoch, losses)


def get_eval_metrics(outputs, labels, run_type, epoch, losses):
    outputs = torch.cat(outputs, dim=0)
    labels = torch.cat(labels, dim=0)
    assert outputs.shape == labels.shape
    # 1. number of correctly predicted tags divided by the total number of tags
    prob_classes = []
    for i in range(labels.size(0)):
        label = labels[i]
        k = label.sum()
        _, idx = outputs[i].topk(k=k)
        predict = torch.zeros_like(outputs[i])
        predict[idx] = 1
        prob_classes.append(predict)
    prob_classes = torch.stack(prob_classes)
    matched_1s = torch.mul(prob_classes, labels)
    correct_tag_percentage = matched_1s.sum() / labels.sum()

    # 2. accuracy if set to 1 if exceeds threshold
    threshold_classes = outputs
    threshold_classes[threshold_classes > args.threshold] = 1
    threshold_classes[threshold_classes <= args.threshold] = 0
    acc = (threshold_classes == labels).sum() / len(threshold_classes.reshape(-1))

    # 3. avg precision
    avg_pre = average_precision_score(labels.detach().numpy(), outputs.detach().numpy(), average='macro')

    # write tensorboard and logging file
    writer.add_scalar("Loss/{}".format(run_type), np.mean(losses), epoch)
    writer.add_scalar("Acc/{}".format(run_type), acc, epoch)
    writer.add_scalar("Pre/{}".format(run_type), avg_pre, epoch)
    writer.add_scalar("Avg_percent/{}".format(run_type), correct_tag_percentage, epoch)
    logging.info("{} - epoch: {}, loss: {}, acc: {}, pre: {}, avg percent: {}".format(
        run_type, epoch, np.mean(losses), acc, avg_pre, correct_tag_percentage))


def get_model():
    n_classes = len(TAGS)
    if args.model =='samplecnn':
        model = SampleCNN(n_classes, config).to(device)
    elif args.model == 'crnn':
        model = CRNN(n_classes, config).to(device)
    elif args.model =='fcn':
        model = FCN(n_classes, config).to(device)
    elif args.model == 'musicnn':
        model = Musicnn(n_classes, config).to(device)
    elif args.model == 'shortchunkcnn_res':
        model = ShortChunkCNN_Res(n_classes, config).to(device)
    elif args.model == 'cnnsa':
        model = CNNSA(n_classes, config).to(device)
    elif args.model == 'transformer':
        model = AutoModelForAudioClassification.from_pretrained(
            "facebook/wav2vec2-base", num_labels=n_classes
        )
    else:
        model = SampleCNN(n_classes, config).to(device)
    return model


# def save_to_onnx(model):
#     dummy_input = torch.randn(1, 96, 4000)
#     torch.onnx.export(model,
#                       dummy_input,
#                       "model/fcn.onnx",
#                       export_params=True,
#                       opset_version=15
#                       )


if __name__ == '__main__':
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    logging.info("Preparing dataset...")
    train_dataset = MyDataset(args.tag_file, args.npy_root, config, "train")
    val_dataset = MyDataset(args.tag_file, args.npy_root, config, "valid")
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size)

    model = get_model()
    # Binary cross-entropy with logits loss combines a Sigmoid layer and the BCELoss in one single class.
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    logging.info("Training and validating model...")
    writer = SummaryWriter('runs/{}_{}_{}'.format(args.model, args.learning_rate, args.batch_size))
    for epoch in range(args.num_epochs):
        train(model, epoch, criterion, optimizer, train_loader)
        validate(model, epoch, criterion, val_loader)

    torch.save(model, 'model/{}.pt'.format(args.model))
    # save_to_onnx(model)
    writer.close()
