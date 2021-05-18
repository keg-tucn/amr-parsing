from torch.utils.tensorboard import SummaryWriter
from utils.tensorboard_utils import save_masked_scores_img_to_tensorboard,\
    save_scores_img_to_tensorboard

LOSS_TEXT = 'loss'
SMATCH_F_SCORE_TEXT = 'smatch_f_score'
SMATCH_PRECISION_TEXT = 'smatch_precision'
SMATCH_RECALL_TEXT = 'smatch_recall'
F_SCORE_TEXT = 'f_score'
ACCURACY_TEXT = 'accuracy'
AMR_LOG_TEXT = 'AMR COMPARISON'
AMR_IMG_SCORES = 'SCORES'
AMR_IMG_RELS = 'GOLD RELATIONS (binary)'


class DataLogger:

    def __init__(self, writer: SummaryWriter, on_training_flow: bool = False):
        self.on_training_flow = on_training_flow
        self.logged_example_index = 0
        self.epoch = 0
        self.writer = writer
        self.loss = 0.0
        self.smatch_f_score = 0.0
        self.smatch_precision = 0.0
        self.smatch_recall = 0.0
        self.f_score = 0.0
        self.accuracy = 0.0
        self.logged_text = ""
        self.input_seq = []
        self.output_seq = []
        self.text_scores = None
        self.color_scores = None
        self.gold_relations = None

    def set_epoch(self, epoch_no):
        self.epoch = epoch_no

    def set_loss(self, loss):
        self.loss = loss

    def set_smatch(self, smatch_f_score, smatch_precision, smatch_recall):
        self.smatch_f_score = smatch_f_score
        self.smatch_precision = smatch_precision
        self.smatch_recall = smatch_recall

    def set_edge_scores(self, f_score, accuracy):
        self.f_score = f_score
        self.accuracy = accuracy

    def set_logged_text(self, logged_text):
        self.logged_text = logged_text

    def set_img_info(self, input_seq, output_seq, text_scores, color_scores, gold_mat):
        self.input_seq = input_seq
        self.output_seq = output_seq
        self.text_scores = text_scores
        self.color_scores = color_scores
        self.gold_relations = gold_mat

    def reset(self):
        self.loss = 0.0
        self.smatch_f_score = 0.0
        self.smatch_precision = 0.0
        self.smatch_recall = 0.0
        self.logged_text = ""
        self.input_seq = []
        self.output_seq = []
        self.text_scores = None
        self.color_scores = None
        self.gold_relations = None


    def to_tensorboard(self):
        self.writer.add_scalar(LOSS_TEXT, self.loss, self.epoch)
        self.writer.add_scalar(SMATCH_F_SCORE_TEXT, self.smatch_f_score, self.epoch)
        self.writer.add_scalar(SMATCH_PRECISION_TEXT, self.smatch_precision, self.epoch)
        self.writer.add_scalar(SMATCH_RECALL_TEXT, self.smatch_recall, self.epoch)
        self.writer.add_scalar(F_SCORE_TEXT, self.f_score, self.epoch)
        self.writer.add_scalar(ACCURACY_TEXT, self.accuracy, self.epoch)
        self.writer.add_text(AMR_LOG_TEXT, self.logged_text, self.epoch)
        save_scores_img_to_tensorboard(self.writer, AMR_IMG_SCORES, self.epoch,
                                       self.input_seq, self.output_seq,
                                       self.text_scores,
                                       on_training_flow=self.on_training_flow)
        save_masked_scores_img_to_tensorboard(self.writer, AMR_IMG_RELS, self.epoch,
                                              self.input_seq, self.output_seq,
                                              self.gold_relations, self.color_scores,
                                              on_training_flow=self.on_training_flow)
        self.reset()
