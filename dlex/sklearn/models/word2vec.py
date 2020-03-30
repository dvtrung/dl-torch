from gensim.sklearn_api import W2VTransformer

from dlex.configs import Params


class Word2Vec(W2VTransformer):
    def __init__(self, params: Params):
        cfg = params.model
        super().__init__(
            size=cfg.dimensions,
            window=cfg.window_size,
            min_count=cfg.min_count or 0,
            sg=1,
            workers=params.train.num_workers or 1,
            iter=params.train.num_epochs,
            seed=cfg.seed or 1
        )