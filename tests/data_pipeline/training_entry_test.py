from absl.testing import absltest
import penman
from penman.models import noop
from penman.surface import (
    Alignment,
    RoleAlignment
)
from data_pipeline.training_entry import TrainingEntry

"""
Tests for data_pipeline/training_entry.py
Run from project dir with 'python -m tests.data_pipeline.training_entry_test'
"""

class TrainingEntryTest(absltest.TestCase):

  def test_construct_from_penman_simple_amr(self):
    """
    # ::id bolt12_07_4800.1
    # ::tok Establishing Models in Industrial Innovation
    (e / establish-01~e.0 
          :ARG1 (m / model~e.1 
                :mod~e.2 (i / innovate-01~e.4 
                      :ARG1 (i2 / industry~e.3))))
    """
    triples = [
      ('e', ':instance', 'establish-01'),
      ('e', ':ARG1', 'm'),
      ('m', ':instance', 'model'),
      ('m', ':mod', 'i'),
      ('i', ':instance', 'innovate-01'),
      ('i', ':ARG1', 'i2'),
      ('i2', ':instance', 'industry')
    ]
    top = 'e'
    epidata = {
      ('e', ':instance', 'establish-01'): [Alignment((0,), prefix='e.')],
      ('e', ':ARG1', 'm'): [],
      ('m', ':instance', 'model'): [Alignment((1,), prefix='e.')],
      ('m', ':mod', 'i'): [],
      ('i', ':instance', 'innovate-01'): [Alignment((4,), prefix='e.')],
      ('i', ':ARG1', 'i2'): [],
      ('i2', ':instance', 'industry'): [Alignment((3,), prefix='e.')]
    }
    g = penman.Graph(triples, top, epidata)
    training_entry = TrainingEntry(
      [], g, unalignment_tolerance=0)
    expected_concepts = [
      'root', 'establish-01', 'model', 'industry', 'innovate-01']
    expected_mat = [
      [None, ':root', None, None, None],
      [None, None, ':ARG1', None, None],
      [None, None, None, None, ':mod'],
      [None, None, None, None, None],
      [None, None, None, ':ARG1', None]
    ]
    self.assertEqual(training_entry.concepts, expected_concepts)
    self.assertEqual(training_entry.adjacency_mat, expected_mat)

  def test_construct_from_penman_reentrancy_amr(self):
    """
    # ::tok This issue should attract the close attention of economists .
    (r / recommend-01~e.2 
          :ARG1 (a / attract-01~e.3 
                :ARG0 (i / issue-02~e.1 
                      :mod (t / this~e.0)) 
                :ARG1 (a2 / attend-02~e.6 
                      :ARG0~e.7 (e / economist~e.8) 
                      :ARG1 i 
                      :ARG1-of (c / close-10~e.5))))
    """
    triples = [
      ('r', ':instance', 'recommend-01'),
      ('r', ':ARG1', 'a'),
      ('a', ':instance', 'attract-01'),
      ('a', ':ARG0', 'i'),
      ('a', ':ARG1', 'a2'),
      ('i', ':instance', 'issue-02'),
      ('i', ':mod', 't'),
      ('t', ':instance', 'this'),
      ('a2', ':instance', 'attend-02'),
      ('a2', ':ARG0', 'e'),
      ('e', ':instance', 'economist'),
      ('a2', ':ARG1', 'i'),
      ('a2', ':ARG1-of', 'c'),
      ('c', ':instance', 'close-10')
    ]
    top = 'r'
    epidata = {
      ('r', ':instance', 'recommend-01'): [Alignment((2,), prefix='e.')],
      ('r', ':ARG1', 'a'): [],
      ('a', ':instance', 'attract-01'): [Alignment((3,), prefix='e.')],
      ('a', ':ARG0', 'i'): [],
      ('a', ':ARG1', 'a2'): [],
      ('i', ':instance', 'issue-02'): [Alignment((1,), prefix='e.')],
      ('i', ':mod', 't'): [],
      ('t', ':instance', 'this'): [Alignment((0,), prefix='e.')],
      ('a2', ':instance', 'attend-02'): [Alignment((6,), prefix='e.')],
      ('a2', ':ARG0', 'e'): [],
      ('e', ':instance', 'economist'): [Alignment((8,), prefix='e.')],
      ('a2', ':ARG1', 'i'): [],
      ('a2', ':ARG1-of', 'c'): [],
      ('c', ':instance', 'close-10'): [Alignment((5,), prefix='e.')]
    }
    g = penman.Graph(triples, top, epidata)
    training_entry = TrainingEntry(
      [], g, unalignment_tolerance=0)
    expected_concepts = [
      'root', 'this', 'issue-02', 'recommend-01', 'attract-01', 'close-10','attend-02', 'economist'
    ]
    expected_mat = [
      [None, None, None, ':root', None, None, None, None],
      [None, None, None, None, None, None, None, None],
      [None, ':mod', None, None, None, None, None, None],
      [None, None, None, None, ':ARG1', None, None, None],
      [None, None, ':ARG0', None, None, None, ':ARG1', None],
      [None, None, None, None, None, None, None, None],
      [None, None, ':ARG1', None, None, ':ARG1-of', None, ':ARG0'],
      [None, None, None, None, None, None, None, None]
    ]
    self.assertEqual(training_entry.concepts, expected_concepts)
    self.assertEqual(training_entry.adjacency_mat, expected_mat)

  def test_construct_from_penman_complex_amr(self):
    """
    Test for the amr:
    What is more they are considered traitors of China , which is a fact of
    cultural tyranny in the cloak of nationalism and patriotism .
    
    (c / consider-01~e.5 
      :ARG1 (p2 / person~e.6 
            :domain~e.1,4,11 (t2 / they~e.3) 
            :ARG0-of~e.6 (b / betray-01~e.6 
                  :ARG1~e.7 (c2 / country~e.8 :wiki "China"~e.8 
                        :name (n2 / name~e.8 :op1 "China"~e.8)))) 
      :mod (m / more~e.2) 
      :mod (t4 / tyrannize-01~e.16
            :ARG2 (c3 / culture~e.15) 
            :ARG1-of (c4 / cloak-01~e.19 
                  :ARG2~e.20 (a / and~e.22 
                        :op1 (n / nationalism~e.21) 
                        :op2 (p / patriotism~e.23)))))"""
    top = 'c'
    triples = [
      ('c', ':instance', 'consider-01'),
      ('c', ':ARG1', 'p2'),
      ('p2', ':instance', 'person'),
      ('p2', ':domain', 't2'),
      ('t2', ':instance', 'they'),
      ('p2', ':ARG0-of', 'b'),
      ('b', ':instance', 'betray-01'),
      ('b', ':ARG1', 'c2'),
      ('c2', ':instance', 'country'),
      ('c2', ':wiki', '"China"'),
      ('c2', ':name', 'n2'),
      ('n2', ':instance', 'name'),
      ('n2', ':op1', '"China"'),
      ('c', ':mod', 'm'),
      ('m', ':instance', 'more'),
      ('c', ':mod', 't4'),
      ('t4', ':instance', 'tyrannize-01'),
      ('t4', ':ARG2', 'c3'),
      ('c3', ':instance', 'culture'),
      ('t4', ':ARG1-of', 'c4'),
      ('c4', ':instance', 'cloak-01'),
      ('c4', ':ARG2', 'a'),
      ('a', ':instance', 'and'),
      ('a', ':op1', 'n'),
      ('n', ':instance', 'nationalism'),
      ('a', ':op2', 'p'),
      ('p', ':instance', 'patriotism')]
    epidata = {
      ('c', ':instance', 'consider-01'): [Alignment((5,), prefix='e.')],
      ('p2', ':instance', 'person'): [Alignment((6,), prefix='e.')],
      ('p2', ':domain', 't2'): [RoleAlignment((1, 4, 11), prefix='e.')],
      ('t2', ':instance', 'they'): [Alignment((3,), prefix='e.')],
      ('p2', ':ARG0-of', 'b'): [RoleAlignment((6,), prefix='e.')],
      ('b', ':instance', 'betray-01'): [Alignment((6,), prefix='e.')],
      ('b', ':ARG1', 'c2'): [RoleAlignment((7,), prefix='e.')],
      ('c2', ':instance', 'country'): [Alignment((8,), prefix='e.')],
      ('c2', ':wiki', '"China"'): [Alignment((8,), prefix='e.')],
      ('c2', ':name', 'n2'): [Alignment((8,), prefix='e.')],
      ('n2', ':instance', 'name'): [Alignment((8,), prefix='e.')],
      ('n2', ':op1', '"China"'): [Alignment((8,), prefix='e.')],
      ('c', ':mod', 'm'): [],
      ('m', ':instance', 'more'): [Alignment((2,), prefix='e.')],
      ('c', ':mod', 't4'): [],
      ('t4', ':instance', 'tyrannize-01'): [Alignment((16,), prefix='e.')],
      ('t4', ':ARG2', 'c3'): [],
      ('c3', ':instance', 'culture'): [Alignment((15,), prefix='e.')],
      ('t4', ':ARG1-of', 'c4'): [],
      ('c4', ':instance', 'cloak-01'): [Alignment((19,), prefix='e.')],
      ('c4', ':ARG2', 'a'): [RoleAlignment((20,), prefix='e.')],
      ('a', ':instance', 'and'): [Alignment((22,), prefix='e.')],
      ('a', ':op1', 'n'): [],
      ('n', ':instance', 'nationalism'): [Alignment((21,), prefix='e.')],
      ('a', ':op2', 'p'): [],
      ('p', ':instance', 'patriotism'): [Alignment((23,), prefix='e.')]}
    g = penman.Graph(triples, top, epidata)
    training_entry = TrainingEntry(
      [], g, unalignment_tolerance=1)
    expected_concepts = [
      'root', 'more', 'they', 'consider-01', 'person', 'betray-01', 'country',
      '"China"', 'name', '"China"', 'culture', 'tyrannize-01', 'cloak-01',
      'nationalism', 'and', 'patriotism']
    expected_mat = [
      [None, None, None, ':root', None, None, None, None, None, None, None, None, None, None, None, None],
      [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
      [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
      [None, ':mod', None, None, ':ARG1', None, None, None, None, None, None, ':mod', None, None, None, None],
      [None, None, ':domain', None, None, ':ARG0-of', None, None, None, None, None, None, None, None, None, None],
      [None, None, None, None, None, None, ':ARG1', None, None, None, None, None, None, None, None, None],
      [None, None, None, None, None, None, None, None, ':name', ':wiki', None, None, None, None, None, None],
      [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
      [None, None, None, None, None, None, None, None, None, ':op1', None, None, None, None, None, None],
      [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
      [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
      [None, None, None, None, None, None, None, None, None, None, ':ARG2', None, ':ARG1-of', None, None, None], 
      [None, None, None, None, None, None, None, None, None, None, None, None, None, None, ':ARG2', None],
      [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
      [None, None, None, None, None, None, None, None, None, None, None, None, None, ':op1', None, ':op2'],
      [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]]
    self.assertEqual(training_entry.concepts, expected_concepts)
    self.assertEqual(training_entry.adjacency_mat, expected_mat)


if __name__ == '__main__':
  absltest.main()