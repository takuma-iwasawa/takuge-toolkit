# -*- coding: utf-8 -*-
from .base import Morph, MorphList
from typing import List, Union
import MeCab
import re
import unicodedata


class MeCabTokenizer:
    def __init__(self, normalize_form='NFKC'):
        self.tagger = MeCab.Tagger()
        self.normalize_form = normalize_form
        self.morphs = None

    def tokenize(self, sentence, extract_element_type: Union[str, List[str]] = 'lemma',
                 extract_pos=None, extract_pos1=None, exclude_pos1=None, remove_eos=True):
        _normed_sentence = unicodedata.normalize(self.normalize_form, str(sentence)) if self.normalize_form else sentence
        _parsed_node_list = self.tagger.parse(_normed_sentence).split('\n')[:-2]
        _parsed_node_list = _parsed_node_list[:-2] if remove_eos else _parsed_node_list[:-1]
        self.morphs = MorphList([MeCabMorph(node) for node in _parsed_node_list])
        if extract_pos is not None:
            self.morphs.filter_morphs_by_pos(extract_pos=extract_pos,
                                             extract_pos1=extract_pos1, exclude_pos1=exclude_pos1)
        return self.morphs.get_concatenated_elements(element_types=extract_element_type)


class MeCabMorph(Morph):
    def __init__(self, node_str):
        super().__init__()
        self.node_str = node_str
        self.pattern = re.compile(',|\t')
        self._set_members()

    def _set_members(self):
        morphemes = self.pattern.split(self.node_str)
        self.surface = morphemes[0]
        self.lemma = morphemes[7] if morphemes[7] != '*' else self.surface
        self.pos = morphemes[1]
        self.pos1 = morphemes[2]
