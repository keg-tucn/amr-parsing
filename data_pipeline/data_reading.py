from typing import List
import re
import os
import definitions

"""Pattern for extracting the id, tokens, alignments and amr from an entry string.
An entry example follows:
# ::id bolt12_07_4800.1 ::amr-annotator SDL-AMR-09 ::preferred
# ::tok Establishing Models in Industrial Innovation
# ::alignments 0-1 1-1.1 2-1.1.1.r 3-1.1.1.1 4-1.1.1
(e / establish-01~e.0 
      :ARG1 (m / model~e.1 
            :mod~e.2 (i / innovate-01~e.4 
                  :ARG1 (i2 / industry~e.3))))
"""
AMR_PATTERN = r'# ::id(.*)\n# ::tok(.*)\n# ::alignments(.*)\n(?=\()(.*)'

def extract_triples(path: str):
  """
  Return a list of (id, tokens, amr string) triples for an AMR data file.
  """
  # Read file data into a string.
  file = open(path,'r')
  text = file.read()
  file.close()
  # Separate the data for each amr (id, token, alignemnts, amr).
  data_blocks = text.split('\n\n')
  triples = []
  for data_block in data_blocks:
    # Use regex to extract the needed info from a text block.
    match = re.search(AMR_PATTERN, data_block, re.DOTALL)
    if match:
      id, sentence, alignments, amr = match.groups()
      triples.append((id, sentence, amr))
  return triples


def get_paths(split: str, subsets: List[str]):
  """
  Returns a list of paths.
  Args:
    split: one of 'training', 'dev', 'test'
    subsets: list of amr data subsets (eg. 'bolt', 'dfa')
  """
  path = os.path.join(definitions.PROJECT_ROOT_DIR,
                      definitions.DATA_PATH,
                      split,
                      definitions.FILE_FORMAT)
  paths = []
  for subset in subsets:
    paths.append(path.format(split, subset))
  return paths