"""LightGBM baseline on the Morgan/RDKit features from models/baselines.py.
Not trained with the epoch loop: one booster per target, early-stopped on validation."""
import warnings

import lightgbm as lgb
import numpy as np
from omegaconf import OmegaConf
from models.baselines import load_features

# LightGBM names the columns itself; sklearn then warns on plain arrays.
warnings.filterwarnings('ignore', message='X does not have valid feature names')


class LightGBMModel:
    def __init__(self, cfg, data_dir, seed):
        self.cfg, self.folder, self.seed = cfg, data_dir / 'features', seed
        self.kwargs = {n: OmegaConf.to_container(cfg[n]) if n in cfg else {} for n in cfg.featurizers}
        self.task, self.boosters = None, []

    def features(self, frame):
        return np.concatenate([load_features(frame, n, self.kwargs[n], self.folder,
                                             self.cfg.feature_workers)
                               for n in self.cfg.featurizers], axis=1)

    def fit(self, train, val, targets, task, out):
        """Fits one booster per target on its labelled rows; returns the best iterations."""
        self.task = task
        Xt, Xv = self.features(train), self.features(val)
        cls = lgb.LGBMRegressor if task == 'regression' else lgb.LGBMClassifier
        best = {}
        for t in targets:
            mt, mv = train[t].notna().to_numpy(), val[t].notna().to_numpy()
            yt, yv = train[t][mt].to_numpy(), val[t][mv].to_numpy()
            if task == 'classification':
                yt, yv = yt.astype(int), yv.astype(int)
            model = cls(**OmegaConf.to_container(self.cfg.params), random_state=self.seed, verbose=-1)
            model.fit(Xt[mt], yt, eval_set=[(Xv[mv], yv)],
                      callbacks=[lgb.early_stopping(self.cfg.early_stopping_rounds, verbose=False)])
            model.booster_.save_model(out / f'{t}.lgbm.txt')
            best[t] = int(model.best_iteration_)
            print(f'{t}: best iteration {best[t]}', flush=True)
            self.boosters.append(model)
        return best

    def predict(self, frame):
        X = self.features(frame)
        cols = [m.predict(X) if self.task == 'regression' else m.predict_proba(X)[:, 1]
                for m in self.boosters]
        values = np.stack(cols, axis=1)
        if not np.isfinite(values).all():
            raise RuntimeError('Nonfinite predictions')
        return values
