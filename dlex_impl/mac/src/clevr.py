import json
import os
from typing import Tuple, List

import h5py
import torch
from PIL import Image
from dlex import MainConfig
from dlex.datasets import DatasetBuilder
from dlex.datasets.nlp.utils import Vocab, write_vocab, nltk_tokenize
from dlex.datasets.torch import Dataset
from dlex.torch import Batch
from dlex.torch.utils.ops_utils import maybe_cuda
from dlex.torch.utils.variable_length_tensor import pad_sequence
from dlex.utils import logger
from torch.utils.data import DataLoader, Dataset as TorchDataset
from torchvision import transforms
from torchvision.models import resnet101, ResNet
from torchvision.transforms import Resize
from tqdm import tqdm
import numpy as np


def tokenize(s: str) -> str:
    return nltk_tokenize(s.lower())


class _CLEVRImage(TorchDataset):
    def __init__(self, root, mode='train'):
        self.root = root
        self.mode = mode
        self.length = len(os.listdir(os.path.join(root, 'images', mode)))
        self.transform = transforms.Compose([
            Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        img = os.path.join(
            self.root, 'images', self.mode,
            'CLEVR_{}_{}.png'.format(self.mode, str(index).zfill(6)))
        img = Image.open(img).convert('RGB')
        return self.transform(img)

    def __len__(self):
        return self.length


class CLEVR(DatasetBuilder):
    def __init__(self, params: MainConfig):
        super().__init__(params, [
            "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip",
            "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip",
            "https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0.zip",
            "https://dl.fbaipublicfiles.com/clevr/CLEVR_CoGenT_v1.0_no_images.zip",
        ])
        self._resnet = None
        self._answer_dict = []

    @property
    def answer_path(self):
        return os.path.join(self.get_processed_data_dir(), "answers.txt")

    @property
    def vocab_path(self):
        return os.path.join(self.get_processed_data_dir(), "vocab.txt")

    @property
    def resnet(self):
        if not self._resnet:
            logger.info("Initializing Resnet...")
            resnet = resnet101(True).cuda()
            resnet.eval()

            def forward(self, x):
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)

                return x

            resnet.forward = forward.__get__(resnet, ResNet)
            self._resnet = resnet
        return self._resnet

    def maybe_preprocess(self, force=False):
        if not super().maybe_preprocess(force):
            return

        for mode in ["train", "val", "test"]:
            with open(os.path.join(
                    self.get_raw_data_dir(), 'CLEVR_v1.0', "questions",
                    f'CLEVR_{mode}_questions.json')) as f:
                data = json.load(f)

            self.process_questions(mode, data)
            # self.process_image_features(mode, len(data['questions']), self.resnet)

    def process_image_features(self, mode, size, resnet):
        logger.info("Extracting image features...")
        batch_size = 50
        dataloader = DataLoader(_CLEVRImage(
            os.path.join(self.get_raw_data_dir(), 'CLEVR_v1.0')),
            batch_size=batch_size,
            num_workers=4)
        with h5py.File(self.get_image_features_path(mode), "w", libver='latest') as f:
            dset = f.create_dataset('data', (size, 1024, 14, 14), dtype='f4')
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            with torch.no_grad():
                for i, img in tqdm(enumerate(dataloader), total=size // batch_size, desc="Extract image features"):
                    img = img.to(device)
                    features = resnet(img).detach().cpu().numpy()
                    dset[i * batch_size:(i + 1) * batch_size] = features

    def process_questions(self, mode, data):
        questions = []
        answers = []
        image_filenames = []

        for question in tqdm(data['questions'], desc=f"Tokenizing {mode}"):
            questions.append(tokenize(question['question']))
            answers.append(question.get('answer', "none").strip())
            image_filenames.append(question['image_filename'])

        if mode == "train":
            self._answer_dict = {ans: i for i, ans in enumerate(set(answers))}
            with open(self.answer_path, "w") as f:
                f.write("\n".join(list(set(answers))))

        with open(self.get_data_path(mode), 'w') as f:
            for q, a, img in zip(questions, answers, image_filenames):
                f.write('\t'.join([
                    img,
                    ' '.join(q),
                    a,
                    str(self._answer_dict.get(a, -1))
                ]) + "\n")

        if mode == "train":
            write_vocab(questions, self.vocab_path)

    def get_pytorch_wrapper(self, mode: str):
        return PytorchCLEVR(self, mode)

    def get_image_features_path(self, mode: str):
        if mode == "valid":
            mode = "val"
        return os.path.join(self.get_processed_data_dir(), f"{mode}_features.hdf5")

    def get_data_path(self, mode: str):
        if mode == "valid":
            mode = "val"
        return os.path.join(self.get_processed_data_dir(), f"{mode}.csv")


class PytorchCLEVR(Dataset):
    def __init__(self, builder, mode: str):
        super().__init__(builder, mode)

        self._vocab = None
        self.h = h5py.File(builder.get_image_features_path(self.mode), 'r')
        self.image_features = self.h['data']

        with open(builder.answer_path) as f:
            self.answers = [ans for ans in f.read().split('\n') if ans.strip() != ""]

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def vocab(self):
        if self._vocab is None:
            self._vocab = Vocab.from_file(self.builder.vocab_path)
        return self._vocab

    @property
    def num_classes(self):
        return len(self.answers)

    @property
    def data(self):
        if self._data is None:
            self._data = []
            with open(self.builder.get_data_path(self.mode)) as f:
                lines = [l for l in f.read().split('\n') if l.strip() != ""]
                logger.info(f"Dataset loaded. Number of samples: {len(lines):,}")

            for line in lines:
                img_path, q, _, a = line.split('\t')
                self._data.append([img_path, self.vocab.encode_token_list(q.split(' ')), int(a)])

            self._data.sort(key=lambda d: len(d[1]))
            logger.info(
                "Question length - max: %d - avg: %d",
                max(len(d[1]) for d in self._data),
                np.average([len(d[1]) for d in self._data]))
        return self._data

    def close(self):
        self.h.close()

    def __getitem__(self, i):
        if type(i) == int:
            _i = slice(i, i + 1)
        else:
            _i = i
        img_path, q, ans = zip(*self.data[_i])
        img = [torch.from_numpy(self.image_features[int(path.rsplit('_', 1)[1][:-4])]) for path in img_path]
        return list(zip(img, q, ans))[0] if type(i) == int else list(zip(img, q, ans))

    def collate_fn(self, batch: List[Tuple]):
        batch = sorted(batch, key=lambda x: len(x[1]), reverse=True)
        imgs, qs, ans = [[b[i] for b in batch] for i in range(3)]
        qs, qlen = pad_sequence(qs, self.vocab.blank_token_idx, True)
        # logger.debug("question length - max: %d, min: %d, avg: %.2f", max(qlen), min(qlen), sum(qlen) / len(qlen))

        return Batch(
            X=(maybe_cuda(torch.stack(imgs)), qs, qlen),
            Y=maybe_cuda(torch.LongTensor(ans)))

