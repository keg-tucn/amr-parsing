from absl.testing import absltest

from data_pipeline.vocab import build_vocab


"""
Tests for data_pipeline/vocab.py
Run from project dir with 'python -m tests.data_pipeline.vocab_test'
"""

class TrainingEntryTest(absltest.TestCase):

  def test_build_vocab_min_freq_1(self):
    words = [
      'Ana', 'are', 'mere', 'cuvinte', 'rosii', 'mere', 'pisica', 'are',
      'portocale', 'are', 'are'
    ]
    special_words = ['PAD', 'UNK']
    min_freq = 1
    vocab = build_vocab(words, special_words, min_freq)
    expected_vocab = {'PAD': 0, 'UNK': 1, 'are': 2, 'mere': 3, 'Ana': 4,
                      'cuvinte': 5, 'rosii': 6, 'pisica': 7, 'portocale': 8}
    self.assertEqual(vocab, expected_vocab)

  def test_build_vocab_min_freq_2(self):
    words = [
      'Ana', 'are', 'mere', 'cuvinte', 'rosii', 'mere', 'pisica', 'are',
      'portocale', 'are', 'are'
    ]
    special_words = ['PAD', 'UNK']
    min_freq = 2
    vocab = build_vocab(words, special_words, min_freq)
    expected_vocab = {'PAD': 0, 'UNK': 1, 'are': 2, 'mere': 3}
    self.assertEqual(vocab, expected_vocab)


if __name__ == '__main__':
  absltest.main()