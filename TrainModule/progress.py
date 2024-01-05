import torch
from tqdm.auto import tqdm

class ProgressBar():
    def __init__(self, task, epoch, initial):
        self.epoch_len = epoch
        # self.train_step_len = task.train_dataloader().__len__()
        self.valid_step_len = task.val_dataloader().__len__()
        if task.test_dataloader():
            self.eval_step_len = task.test_dataloader().__len__()
        else: self.eval_step_len = 0

        # self.total_step_len = self.train_step_len + self.valid_step_len
        self.cur_step = initial
        self.loss = 0

        self.Training_bar = tqdm(
                    initial = self.cur_step,
                    leave = False,
                    desc = 'Training steps {}'.format(self.cur_step),
                    total = self.epoch_len,
                    postfix = {'loss': self.loss},
                    dynamic_ncols = True
                         )
        self.Validation_bar = tqdm(
                    leave = False,
                    desc = 'On validating',
                    total = self.valid_step_len,
                    dynamic_ncols = True,
                    postfix = {'loss': self.loss}
                         )
        # self.Evaluation_bar = tqdm(
        #             leave = False,
        #             desc = 'Evaluating',
        #             total = self.eval_step_len,
        #             dynamic_ncols = True,
        #                  )

    def on_training_start(self):
        self.cur_step += 1
        self.Training_bar.display()

    def on_training_end(self, output):
        if type(output) in [list, tuple]:
            self.loss = [round(op.detach().item(), 4) for op in output]
        else:
            self.loss = output.detach().item()
        self.Training_bar.update(1)
        self.Training_bar.set_description('Training step {}'.format(self.cur_step))
        self.Training_bar.set_postfix({'loss': self.loss})

    def on_train_batch_end(self, output):
        self.loss = output.detach().item()



    def on_validation_start(self):

        self.Validation_bar.set_description('Validation step {}'.format(self.cur_step))
        self.Validation_bar.display()

    def on_validation_batch_end(self, output):
        self.loss = 0
        # self.loss = output['loss'].detach().item()
        self.Validation_bar.update(1)
        self.Validation_bar.set_postfix({'loss': self.loss})

    def val_bar_reset(self):
        self.Validation_bar.refresh()
        self.Validation_bar.reset()
