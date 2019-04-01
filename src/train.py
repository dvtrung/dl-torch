import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs import Configs
from utils.model_utils import get_model, get_dataset
from utils.logging import logger
from utils.ops_utils import Tensor, LongTensor

def train(params):
    Dataset = get_dataset(params)
    Model = get_model(params)

    Dataset.prepare()

    if args.debug:
        dataset_train = Dataset("debug", params)
        dataset_test = Dataset("debug", params)
    else:
        dataset_train = Dataset("train", params)
        dataset_test = Dataset("test", params)
    model = Model(params, dataset_train)
    model.cuda()

    logger.info("Train size: %d" % len(dataset_train))

    data_train = DataLoader(
        dataset_train,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
        collate_fn=dataset_train.collate_fn)

    data_test = DataLoader(
        dataset_test,
        batch_size=params.batch_size,
        collate_fn=dataset_test.collate_fn)

    logger.info("Training model...")
    logger.info(torch.cuda.get_device_name(0))

    optim = torch.optim.Adam(model.parameters(), lr=params.optimizer.learning_rate)
    epoch = 0
    for ei in range(epoch + 1, epoch + params.num_epochs + 1):
        loss_sum = 0

        for item in tqdm(data_train, desc="Epoch %d" % ei):
            wx, y = item['wtokens'], item['wtags']

            model.zero_grad()
            loss = torch.mean(model(None, wx, y))  # forward pass and compute loss
            loss.backward()  # compute gradients
            optim.step()  # update parameters
            loss = loss.item()
            loss_sum += loss

        total = 0

        acc = 0.
        for item in data_test:
            wx, y = item['wtokens'], item['wtags']
            y_pred = model.decode(None, wx)
            #print(y_pred[0])
            #print(y_test.numpy()[0])
            #print()

            for pr, gt in zip(y_pred, y):
                acc += dataset_test.eval(pr, gt)
            total += len(y_pred)
            #print(y_pred.shape, y_test.numpy()[:, 1:].shape)
            #print(y_pred[20])
            #print(y_test.numpy()[20])

        loss_sum /= len(data_train)

        print(loss_sum, total, acc / total)


if __name__ == "__main__":
    configs = Configs()
    configs.parse_args()
    configs.get_params()
    params = configs.params
    args = configs.args

    torch.manual_seed(params.seed)

    train(params)
