from methods.replay import Replay

class Joint(Replay):
    def __init__(self, model, tasks, budget, m_update, device):
        super().__init__(model, tasks, budget, m_update, device)

    def memorize(self, task, budgets):
        return task