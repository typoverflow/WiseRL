import torch


class EarlyStopManager():
    def __init__(
        self,
        tolerance: int,
        mode: str = "min"
    ):
        assert mode in {"min", "max"}, f"EarlyStopManager: metric should be either max or min"
        self.tolerance = tolerance
        self.mode = mode
        self.best = None
        self.best_metric = None
        self.counter = 0

        self.reset()

    def reset(self):
        self.best = None
        self.best_metrics = 9e9 if self.mode == "min" else -9e9
        self.counter = 0

    def step(self, model, metric):
        if self.best is None:
            self.best = model.state_dict()
            self.best_metric = metric
            self.counter = 0
        elif self.mode == "min" and metric < self.best_metric:
            self.best = model.state_dict()
            self.best_metric = metric
            self.counter = 0
        elif self.mode == "max" and metric > self.best_metric:
            self.best = model.state_dict()
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.tolerance:
            should_stop = True
        else:
            should_stop = False

        return should_stop, self.best, self.best_metric
