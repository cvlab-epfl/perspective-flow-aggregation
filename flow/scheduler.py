import torch

from math import cos, pi, tanh

def anneal_linear(start, end, proportion, params=None):
    return start + proportion * (end - start)

def anneal_multi_steps(start, end, proportion, params):
    steps = params['steps']
    gamma = params['gamma']
    lr = start
    for s in steps:
        if proportion >= s:
            lr *= gamma
    return lr

def anneal_cos(start, end, proportion, params=None):
    cos_val = cos(pi * proportion) + 1

    return end + (start - end) / 2 * cos_val

def anneal_cospow(start, end, proportion, params=None):
    power = 5

    cos_val = 0.5 * (cos(pi * proportion) + 1) + 1
    cos_val = power ** cos_val - power
    cos_val = cos_val / (power ** 2 - power)

    return end + (start - end) * cos_val

def anneal_poly(start, end, proportion, power=0.9):
    return (start - end) * (1 - proportion) ** power + end

def anneal_tanh(start, end, proportion, lower=-6, upper=3):
    return end + (start - end) / 2 * (1 - tanh(lower + (upper - lower) * proportion))

class Phase:
    def __init__(self, start, end, n_iter, anneal_fn, params = None):
        self.start, self.end = start, end
        self.n_iter = n_iter
        self.anneal_fn = anneal_fn
        self.params = params
        self.n = 0

    def step(self):
        self.n += 1

        return self.anneal_fn(self.start, self.end, self.n / self.n_iter, self.params)

    def reset(self):
        self.n = 0

    @property
    def is_done(self):
        return self.n >= self.n_iter

class WarmupScheduler:
    def __init__(
        self, 
        optimizer, 
        lr_base, 
        max_iter,
        strategy, 
        params,
        warmup_iter = 1000
    ):
        self.optimizer = optimizer

        # phase_map = {
        #     'linear': anneal_linear,
        #     'cos': anneal_cos,
        #     'cospow': anneal_cospow,
        #     'poly': anneal_poly,
        #     'tanh': anneal_tanh,
        # }

        warmup_lr_start = 0
        ph1 = Phase(warmup_lr_start, lr_base, warmup_iter, anneal_linear)

        if strategy == 'linear':
            lr_end = params['lr_end']
            ph2 = Phase(lr_base, lr_end, max_iter - warmup_iter, anneal_linear)

        elif strategy == 'multi_steps':
            steps = [(s - warmup_iter)/(max_iter - warmup_iter) for s in params['steps']]
            gamma = params['gamma']
            tmp_params = {'steps': steps, 'gamma': gamma}
            ph2 = Phase(lr_base, None, max_iter - warmup_iter, anneal_multi_steps, tmp_params)

        elif strategy == 'cosine':
            lr_end = params['lr_end']
            ph2 = Phase(lr_base, lr_end, max_iter - warmup_iter, anneal_cos)
            
        else:
            print('Not supported scheduler strategy "%s"' % strategy)
            assert(0)

        self.lr_phase = [ph1, ph2]

        # self.lr_phase = [
        #     Phase(lr_min, lr_max, phase1, phase_map[phase[0]]),
        #     Phase(lr_max, lr_min / 1e4, phase2, phase_map[phase[1]]),
        # ]

        self.phase = 0

    def step(self):
        lr = self.lr_phase[self.phase].step()
        if self.lr_phase[self.phase].is_done:
            self.phase += 1
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr

    def step_multiple(self, num_step):
        lr = 0
        for i in range(num_step):
            lr = self.lr_phase[self.phase].step()
            if self.lr_phase[self.phase].is_done:
                self.phase += 1
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr

if __name__ == "__main__":
    lr_base = 1
    max_iter = 10000

    model = torch.nn.Linear(10, 2) # just a test module
    optimizer = torch.optim.SGD(model.parameters(), lr=lr_base)
    # scheduler = WarmupScheduler(optimizer, lr_base, max_iter,  'linear', {'lr_end': lr_base*0.3})
    # scheduler = WarmupScheduler(optimizer, lr_base, max_iter, 'multi_steps', {'steps': (7000, 8000), 'gamma': 0.1})
    scheduler = WarmupScheduler(optimizer, lr_base, max_iter,  'cosine', {'lr_end': lr_base*0.01})
    for i in range(max_iter):
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(current_lr)