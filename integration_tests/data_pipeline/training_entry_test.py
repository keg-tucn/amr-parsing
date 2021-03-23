from absl.testing import absltest
import penman
from penman.models import noop
from penman.surface import (
    Alignment,
    RoleAlignment
)
from data_pipeline.training_entry import TrainingEntry


class TrainingEntryTest(absltest.TestCase):

    def test_training_entry_amr_with_i(self):
        sentence = 'Am I being foolish in doing this ?'
        sentence = sentence.split()
        amr_str = """
    (f / foolish~e.3 :mode~e.7 interrogative~e.7 
      :domain~e.0,2 (i / i~e.1) 
      :condition~e.4 (d / do-02~e.5 
            :ARG0 i 
            :ARG1 (t / this~e.6)))
    """
        g = penman.decode(amr_str, model=noop.model)
        training_entry = TrainingEntry(sentence, g)
        # TODO: test training entry contents.
        self.assertNotEqual(training_entry, None)

    def test_training_entry_with_i_2(self):
        sentence = 'Ultimately , it will reveal its wolf nature , and I hope the sheep will not be fooled by it .'
        amr_str = """
    (a / and~e.9 
      :op1 (r / reveal-01~e.4 
            :ARG0 (i / it~e.2) 
            :ARG1 (n / nature~e.7 
                  :mod (w / wolf~e.6) 
                  :poss~e.5 i~e.5) 
            :time (u / ultimate~e.0)) 
      :op2 (h / hope-01~e.11 
            :ARG0 (i2 / i~e.10) 
            :ARG1 (f / fool-01~e.17 :polarity~e.15 -~e.15 
                  :ARG0~e.18 i~e.19 
                  :ARG1 (s / sheep~e.13))))
    """
        sentence = sentence.split()
        g = penman.decode(amr_str)
        training_entry = TrainingEntry(sentence, g)
        # TODO: test training entry contents.
        self.assertNotEqual(training_entry, None)

    def test_training_entry_amr_one_node(self):
        sentence = '%pw'
        amr_str = """
    (t / thing)
    """
        g = penman.decode(amr_str)
        training_entry = TrainingEntry(sentence, g, unalignment_tolerance=1)
        self.assertNotEqual(training_entry, None)


if __name__ == '__main__':
    absltest.main()
