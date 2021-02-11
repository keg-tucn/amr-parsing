from typing import List
import re
import os
import definitions

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
    pattern = r'# ::id(.*)\n# ::tok(.*)\n# ::alignments(.*)\n(?=\()(.*)'
    match = re.search(pattern, data_block, re.DOTALL)
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