from methods.replay import Replay

class Joint(Replay):
    def __init__(self, model, tasks, budget, m_update, device, focal_gamma, pseudo_label, retrain):
        super().__init__(model, tasks, budget, m_update, device, focal_gamma, pseudo_label, retrain)

    def memorize(self, task, budgets):
        if self.pseudo_label:
            task.train_mask[:] = True
            task.y = task.pseudo_labels
            return task
        else:
            return task