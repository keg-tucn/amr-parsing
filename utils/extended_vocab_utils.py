from copy import deepcopy

import torch

from data_pipeline.dataset import UNK


def construct_extended_vocabulary(unnumericalized_inputs, vocabs):
    """
     Computes the extended vocab of the already existing vocabulary
    and the words in the sentences that are in the batch

    Args:
       vocabs: the vocabulary containing the concepts and tokens
       extended_vocab: composed of the vocabulary and the
              tokens in each sentence of the batch
       extended_vocab_size: the size of the extended vocabulary
    """
    extended_vocab = deepcopy(vocabs.shared_vocab)

    # compute extended vocabulary size
    extended_vocab_size = len(extended_vocab.items())

    # add in the extended vocabulary the words from the initial input
    for sentence in unnumericalized_inputs:
        for token in sentence:
            if token not in extended_vocab.keys():
                extended_vocab[token] = extended_vocab_size
                extended_vocab_size += 1
    return extended_vocab, extended_vocab_size


def numericalize_concepts(extended_vocab, unnumericalized_concepts):
    """
       Numericalize the concepts after the new extended vocabulary

      Args:
         extended_vocab: the vocabulary after which the concepts will be numericalized
         unnumericalized_concepts: the concepts that need to be numericalized
         extended_vocab_size: the size of the extended vocabulary
       Returns:
         gold_outputs: the tensor with the numericalized concepts
    """

    # numericalize the concepts as a list
    gold_outputs_list = [[extended_vocab[word] if word in extended_vocab.keys() else extended_vocab[UNK]
                          for word in sentence] for sentence in unnumericalized_concepts]

    # create tensor from list
    gold_outputs_tensor = torch.as_tensor(gold_outputs_list)

    # transpose the tensor to have the desired shape
    gold_outputs = torch.transpose(gold_outputs_tensor, 0, 1)
    return gold_outputs