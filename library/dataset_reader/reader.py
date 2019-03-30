from overrides import overrides

from typing import Iterator, List, Dict
import numpy as np
import torch
import torch.optim as optim
from allennlp.data.dataset_readers import DatasetReader
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField, SequenceLabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenCharactersIndexer
from allennlp.data.tokenizers import Token


import glob
import unicodedata
import string
import random


@DatasetReader.register("name-reader")
class Reader(DatasetReader):
    
    def __init__(self , token_indexers: Dict[str, SingleIdTokenIndexer] = None, token_character_indexer: Dict[str, TokenCharactersIndexer] = None) -> None:
        super().__init__(lazy=False)
        self.token_indexers = token_indexers or{"tokens": SingleIdTokenIndexer()}
        self.token_character_indexers = token_character_indexer or {"token_characters": TokenCharactersIndexer()}
        self.all_letters = string.ascii_letters + " .,;'"
        self.category_lines = {}
        self.all_categories = []
    @overrides
    def text_to_instance(self, names: List[str], categories: List[str] = None) -> Instance:
        tokens = [Token(name) for name in names]
        token_field = TextField(tokens, self.token_indexers)
        fields = {"tokens": token_field}
        token_character_field = TextField(tokens, self.token_character_indexers)
        fields["token_characters"] = token_character_field
        if categories != None:
            label_field = SequenceLabelField(labels=categories, sequence_field = token_field)
            fields["labels"] = label_field

        return Instance(fields)
    def unicode_to_ascii(self, s: str) -> str:
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in self.all_letters
        )
    def readLines(self, filename:str) -> list:
        lines = open(filename).read().strip().split('\n')
        return [self.unicode_to_ascii(line) for line in lines]
    @overrides
    def _read(self, file_path: str) -> Iterator[Instance]:
        all_filenames = glob.glob(file_path)
        name_cats = []
        for filename in all_filenames:
            category = filename.split('/')[-1].split('.')[0]
            lines = self.readLines(filename)
            self.category_lines[category] = lines
            name_cats.extend([(word, category) for word in lines])
        random.shuffle(name_cats)
        for i in range(0, len(name_cats), 10):
            chunk = name_cats[i:i + 10]
            yield self.text_to_instance([pair[0] for pair in chunk], [pair[1] for pair in chunk])
