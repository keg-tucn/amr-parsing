import numpy as np

from absl.testing import absltest
from torch.utils.tensorboard import SummaryWriter

from utils.tensorboard_utils import save_scores_img_to_tensorboard


class TensorboardUtilsTest(absltest.TestCase):
    def test_create_img(self):
        input_seq = ['dog', 'eat-01', 'bone']
        output_seq = ['dog', 'read-01']
        scores = [
            [1, 0],
            [0.1, 0.5],
            [0.2, 0]
        ]
        scores = np.array(scores)
        writer = SummaryWriter("./test_img")
        writer.add_text("AMR comparison", "The dog eats a bone", 0)
        writer.add_scalar("loss", 1.0, 0)
        writer.add_scalar("loss", 1.5, 1)
        save_scores_img_to_tensorboard(writer, "Test", 0, input_seq, output_seq, scores)
        writer.close()
        pass


if __name__ == '__main__':
    absltest.main()
