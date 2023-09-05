from methods.replay import Replay

class Joint(Replay):
    def __init__(self, model, tasks, device):
        super().__init__(model, tasks, device)

    def memorize(self, task):
        self.memory_bank = [task]