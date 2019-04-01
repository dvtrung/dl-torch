import os
import torch
# import curses
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from configs import Configs
from utils.model_utils import get_model, get_dataset, get_optimizer, load_checkpoint, save_checkpoint
from utils.logging import logger, set_log_dir
from utils.ops_utils import Tensor, LongTensor

def train(params, args):
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

    if torch.cuda.is_available():
        logger.info("Cuda available: " + torch.cuda.get_device_name(0))
        model.cuda()

    optim = get_optimizer(params, model)

    if args.load:
        load_checkpoint(args.load, params, model, optim)
        logger.info("Saved model loaded: %s" % args.load)
        logger.info("Epoch: %f" % (model.global_step / len(dataset_train)))
        res = eval(model, dataset_test, params)
        logger.info("Evaluate saved model: %f" % res)

    logger.info("Train size: %d" % len(dataset_train))

    data_train = DataLoader(
        dataset_train,
        batch_size=params.batch_size,
        shuffle=params.shuffle,
        collate_fn=dataset_train.collate_fn)

    logger.info("Training model...")

    epoch = int(model.global_step / len(dataset_train))
    for ei in range(epoch + 1, epoch + params.num_epochs + 1):
        logger.info("--- Epoch %d ---" % ei)
        loss_sum = 0

        epoch_step = 0
        for id, batch in enumerate(tqdm(data_train, desc="Epoch %d" % ei)):
            model.zero_grad()
            loss = model.loss(batch)
            loss.backward()
            optim.step()
            loss_sum += loss.item()

            epoch_step += 1
            if args.debug and epoch_step > 10:
                break

            model.global_step = (ei - 1) * len(dataset_train) + id * params.batch_size

        res = eval(model, dataset_test, params)

        logger.info("Loss: %f, Acc: %f" % (loss, res))
        save_checkpoint("epoch-%d" % ei, params, model, optim)

def eval(model, dataset, params):
    data_loader = DataLoader(
        dataset,
        batch_size=params.test_batch_size or params.batch_size,
        collate_fn=dataset.collate_fn)

    total = 0
    acc = 0.
    for batch in tqdm(data_loader, desc="Eval"):
        y_pred = model.predict(batch)
        for pr, gt in zip(y_pred, batch['wtags']):
            # logger.info("%d %d" % (pr, gt))
            acc += dataset.eval(pr, gt)
        total += len(y_pred)

    return acc / total

def main():
    configs = Configs()
    configs.parse_args()
    configs.get_params()
    params = configs.params
    args = configs.args

    torch.manual_seed(params.seed)

    log_dir = os.path.join("logs", args.config_path)
    os.makedirs(log_dir, exist_ok=True)
    set_log_dir(os.path.join(log_dir, datetime.now().strftime('%Y%m%d-%H%M%S')))

    train(params, args)

def draw_screen(stdscr):
    stdscr.clear()
    stdscr.refresh()

    k = 0
    cursor_x = 0
    cursor_y = 0
    while (k != ord('q')):
        stdscr.clear()
        height, width = stdscr.getmaxyx()

        if k == curses.KEY_DOWN:
            cursor_y = cursor_y + 1
        elif k == curses.KEY_UP:
            cursor_y = cursor_y - 1
        elif k == curses.KEY_RIGHT:
            cursor_x = cursor_x + 1
        elif k == curses.KEY_LEFT:
            cursor_x = cursor_x - 1

        cursor_x = max(0, cursor_x)
        cursor_x = min(width-1, cursor_x)

        cursor_y = max(0, cursor_y)
        cursor_y = min(height-1, cursor_y)

        # Declaration of strings
        title = "Curses example"[:width-1]
        subtitle = "Written by Clay McLeod"[:width-1]
        keystr = "Last key pressed: {}".format(k)[:width-1]
        statusbarstr = "Press 'q' to exit | STATUS BAR | Pos: {}, {}".format(cursor_x, cursor_y)
        if k == 0:
            keystr = "No key press detected..."[:width-1]

        # Centering calculations
        start_x_title = int((width // 2) - (len(title) // 2) - len(title) % 2)
        start_x_subtitle = int((width // 2) - (len(subtitle) // 2) - len(subtitle) % 2)
        start_x_keystr = int((width // 2) - (len(keystr) // 2) - len(keystr) % 2)
        start_y = int((height // 2) - 2)

        # Rendering some text
        whstr = "Width: {}, Height: {}".format(width, height)
        stdscr.addstr(0, 0, whstr, curses.color_pair(1))

        # Render status bar
        stdscr.attron(curses.color_pair(3))
        stdscr.addstr(height-1, 0, statusbarstr)
        stdscr.addstr(height-1, len(statusbarstr), " " * (width - len(statusbarstr) - 1))
        stdscr.attroff(curses.color_pair(3))

        # Turning on attributes for title
        stdscr.attron(curses.color_pair(2))
        stdscr.attron(curses.A_BOLD)

        # Rendering title
        stdscr.addstr(start_y, start_x_title, title)

        # Turning off attributes for title
        stdscr.attroff(curses.color_pair(2))
        stdscr.attroff(curses.A_BOLD)

        # Print rest of text
        stdscr.addstr(start_y + 1, start_x_subtitle, subtitle)
        stdscr.addstr(start_y + 3, (width // 2) - 2, '-' * 4)
        stdscr.addstr(start_y + 5, start_x_keystr, keystr)
        stdscr.move(cursor_y, cursor_x)

        # Refresh the screen
        stdscr.refresh()

        # Wait for next input
        k = stdscr.getch()


    #stdscr.addstr("test")
    #stdscr.refresh()



if __name__ == "__main__":
    # curses.wrapper(draw_screen)
    main()
