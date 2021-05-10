from torch.utils.tensorboard import SummaryWriter
from utils.tensorboard_utils import save_scores_img_to_tensorboard

LOSS_TEXT = 'loss'
SMATCH_F_SCORE_TEXT = 'smatch_f_score'
SMATCH_PRECISION_TEXT = 'smatch_precision'
SMATCH_RECALL_TEXT = 'smatch_recall'
AMR_LOG_TEXT = 'AMR COMPARISON'
AMR_IMG_SCORES = 'AMR SCORES COMPARISON'
AMR_IMG_PREDS = 'AMR PREDICTIONS COMPARISON'
AMR_IMG_RELS = 'AMR GOLD RELS COMPARISON'


class LoggedData:

    def __init__(self, train_writer: SummaryWriter, eval_writer: SummaryWriter):
        self.logged_example_index = 0
        self.train_writer = train_writer
        self.eval_writer = eval_writer
        self.epoch = 0
        self.epoch_loss = 0.0
        self.dev_loss = 0.0
        self.smatch_f_score = 0.0
        self.smatch_precision = 0.0
        self.smatch_recall = 0.0
        self.logged_text = ""
        self.input_seq = []
        self.output_seq = []
        self.scores = None
        self.predictions = None
        self.gold_relations = None

    def set_epoch(self, epoch_no):
        self.epoch = epoch_no

    def set_losses(self, epoch_loss, dev_loss):
        self.epoch_loss = epoch_loss
        self.dev_loss = dev_loss

    def set_smatch(self, smatch_f_score, smatch_precision, smatch_recall):
        self.smatch_f_score = smatch_f_score
        self.smatch_precision = smatch_precision
        self.smatch_recall = smatch_recall

    def set_logged_text(self, logged_text):
        self.logged_text = logged_text

    def set_img_info(self, input_seq, output_seq, scores, predictions, gold_mat):
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.scores = scores
        self.predictions = predictions
        self.gold_relations = gold_mat

    def write_to_tensorboard(self):
        self.train_writer.add_scalar(LOSS_TEXT, self.epoch_loss, self.epoch)
        self.eval_writer.add_scalar(LOSS_TEXT, self.dev_loss, self.epoch)
        self.eval_writer.add_scalar(SMATCH_F_SCORE_TEXT, self.smatch_f_score, self.epoch)
        self.eval_writer.add_scalar(SMATCH_PRECISION_TEXT, self.smatch_precision, self.epoch)
        self.eval_writer.add_scalar(SMATCH_RECALL_TEXT, self.smatch_recall, self.epoch)
        self.eval_writer.add_text(AMR_LOG_TEXT, self.logged_text, self.epoch)
        save_scores_img_to_tensorboard(self.eval_writer, AMR_IMG_SCORES, self.epoch,
                                       self.input_seq, self.output_seq, self.scores)
        save_scores_img_to_tensorboard(self.eval_writer, AMR_IMG_PREDS, self.epoch,
                                       self.input_seq, self.output_seq, self.predictions)
        save_scores_img_to_tensorboard(self.eval_writer, AMR_IMG_RELS, self.epoch,
                                       self.input_seq, self.output_seq, self.gold_relations)
