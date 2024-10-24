import random
import torch

class Ema(object):
    def __init__(self, source, target, decay=0.9999, start_iter=0):
        self.source = source
        self.target = target
        self.decay = decay
        self.start_iter = start_iter
        self.source_dict = self.source.state_dict()
        self.target_dict = self.target.state_dict()
        print("Initialize the copied generator's parameters to be source parameters.")
        with torch.no_grad():
            for p_ema, p in zip(self.target.parameters(), self.source.parameters()):
                p_ema.copy_(p)
            for b_ema, b in zip(self.target.buffers(), self.source.buffers()):
                b_ema.copy_(b)

    def update(self, iter=None):
        if iter >= 0 and iter < self.start_iter:
            decay = 0.0
        else:
            decay = self.decay

        with torch.no_grad():
            for p_ema, p in zip(self.target.parameters(), self.source.parameters()):
                p_ema.copy_(p.lerp(p_ema, decay))
            for (b_ema_name, b_ema), (b_name, b) in zip(self.target.named_buffers(), self.source.named_buffers()):
                if "num_batches_tracked" in b_ema_name:
                    b_ema.copy_(b)
                else:
                    b_ema.copy_(b.lerp(b_ema, decay))
