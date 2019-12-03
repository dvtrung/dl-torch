import os

from dlex.datasets.builder import DatasetBuilder


class ImageDataset(DatasetBuilder):
    def format_output(self, y_pred, batch_item, tag="default") -> (str, str, str):
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
            raise ValueError("Unknown output format.")

    @property
    def num_channels(self):
        """
        :return: number of channels in the input image
        """
        raise NotImplementedError

    @property
    def input_shape(self):
        """
        :return: shape of the input image
        """
        raise NotImplementedError