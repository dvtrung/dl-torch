import abc
import os

from dlex.datasets.builder import DatasetBuilder


class ImageDatasetBuilder(DatasetBuilder):
    @abc.abstractmethod
    def evaluate(self, hypothesis, reference, metric: str) -> (int, int):
        if metric == 'acc':
            return int(hypothesis == reference), 1
        else:
            super().evaluate(hypothesis, reference, metric)

    def format_output(self, y_pred, batch_item, tag="default") -> (str, str, str):
        y_pred = y_pred.cpu().detach().numpy()
        format = self.params.dataset.output_format
        if format is None or format == "default":
            return "", str(batch_item.Y), str(y_pred)
        elif format == "img":
            plt.subplot(1, 2, 1)
            plt.imshow(self.to_img(batch_item[0].cpu().detach().numpy()))
            plt.subplot(1, 2, 2)
            plt.imshow(self.to_img(y_pred))
            fn = os.path.join(self.params.output_dir, 'infer-%s.png' % tag)
            plt.savefig(fn)
            return "file: %s" % fn
        else:
            raise Exception("Unknown output format.")