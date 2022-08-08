# -*- coding: utf-8 -*-
from .base import BaseMorph, MorphList
from typing import List, Dict, Iterable, Union, Optional
import MeCab
import re
import unicodedata
import regex
from itertools import pairwise
from copy import copy


class MeCabTokenizer:
    def __init__(self, tagger_option: Optional[str] = None, normalize_form='NFKC', normalize_digit=False,
                 stopwords: Optional[List[str]] = None):
        self.tagger = MeCab.Tagger(tagger_option) if tagger_option else MeCab.Tagger()
        self.normalize_form = normalize_form
        self.normalize_digit = normalize_digit
        self.stopwords = stopwords
        self.parse_mecab_pattern = re.compile(r',|\t')
        self.katakana_pattern = regex.compile(r'[\p{Script=Katakana}ー]+')

    def tokenize(self, sentence, extract_element_type: Union[str, List[str]] = 'lemma',
                 extract_pos=None, extract_pos1=None, exclude_pos1=None, exclude_pos2=None, remove_eos=True,
                 concat_suffix_target: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
                 concat_suffix_like_noun_target: Optional[List[str]] = None,
                 concat_prefix_target: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
                 concat_pos_targets: Optional[Dict[str, List[str]]] = None,
                 max_concat_nouns: Union[bool, int] = False, concat_nouns_type: List[str] = None,
                 concat_katakana_nouns=False) -> List[str]:
        parsed_node_list = self.tokenize_as_mecab_format(sentence=sentence, remove_eos=remove_eos)
        return self.extract_words_from_mecab_format(
            parsed_node_list=parsed_node_list, extract_element_type=extract_element_type, extract_pos=extract_pos,
            extract_pos1=extract_pos1, exclude_pos1=exclude_pos1, exclude_pos2=exclude_pos2,
            concat_suffix_target=concat_suffix_target, concat_suffix_like_noun_target=concat_suffix_like_noun_target,
            concat_prefix_target=concat_prefix_target, concat_katakana_nouns=concat_katakana_nouns,
            max_concat_nouns=max_concat_nouns, concat_nouns_type=concat_nouns_type,
        )

    def tokenize_as_mecab_format(self, sentence: str, remove_eos=True):
        _normed_sentence = sentence
        if self.normalize_form:
            _normed_sentence = unicodedata.normalize(self.normalize_form, str(sentence))

        if self.normalize_digit:
            pattern = re.compile(r'\d+')
            _normed_sentence = pattern.sub('0', str(sentence))

        parsed_node_list = self.tagger.parse(_normed_sentence).split('\n')[:-1]
        return parsed_node_list[:-1] if remove_eos else parsed_node_list

    def extract_words_from_mecab_format(self, parsed_node_list, extract_element_type: Union[str, List[str]] = 'lemma',
                                        extract_pos=None, extract_pos1=None, exclude_pos1=None, exclude_pos2=None,
                                        concat_suffix_target: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
                                        concat_suffix_like_noun_target: Optional[List[str]] = None,
                                        concat_prefix_target: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
                                        max_concat_nouns: Union[bool, int] = False, concat_nouns_type: List[str] = None,
                                        concat_katakana_nouns=False) -> List[str]:
        morph_attrs = ['surface', 'pos', 'pos1', 'pos2', 'pos3', 'ctype', 'cform', 'lemma', 'pronounce']
        morphs = MeCabMorphList([
            MeCabMorph(dict(zip(morph_attrs, self.parse_mecab_pattern.split(node))))
            for node in parsed_node_list
        ])

        if max_concat_nouns:
            morphs.concat_nouns(max_concats=max_concat_nouns, concat_nouns_type=concat_nouns_type)

        if concat_katakana_nouns:
            morphs.concat_katakana_nouns(pattern=self.katakana_pattern)

        if concat_suffix_target:
            if type(concat_suffix_target) != list:
                concat_suffix_target = [concat_suffix_target]
            for target in concat_suffix_target:
                morphs.concat_suffix(**target)

        if concat_suffix_like_noun_target:
            morphs.concat_suffix_like_noun(target_suffixes=concat_suffix_like_noun_target)

        if concat_prefix_target:
            if type(concat_prefix_target) != list:
                concat_prefix_target = [concat_prefix_target]
            for target in concat_prefix_target:
                morphs.concat_suffix(**target)

        if self.stopwords:
            morphs.remove_stopwords(self.stopwords)

        morphs.filter_morphs_by_pos(extract_pos=extract_pos, extract_pos1=extract_pos1,
                                    exclude_pos1=exclude_pos1, exclude_pos2=exclude_pos2)

        return morphs.get_concatenated_elements(element_types=extract_element_type)


class MeCabMorph(BaseMorph):
    pos3: str
    """品詞細分類3"""

    ctype: str
    """活用型"""

    cform: str
    """活用形"""

    pronounce: str
    """発音"""

    def __post_init__(self):
        for attr in self.__dict__.keys():
            if self.__getattribute__(attr) == "*":
                setattr(self, attr, None)


class MeCabMorphList(MorphList):
    def __init__(self, morphs_list: Iterable[MeCabMorph]):
        super().__init__(morphs_list=morphs_list)

    def concat_suffix(self, target_pos: Optional[str] = None, target_regex: Optional[str] = None,
                      suffix_pos2: Optional[Union[str, List[str]]] = None):
        if '接尾' not in [morph.pos1 for morph in self.morphs_list] or len(self.morphs_list) < 2:
            return
        else:
            modified_morphs_list = []
            is_concat = False
            morph_modified = self.morphs_list[0]
            for morph1, morph2 in pairwise(self.morphs_list):
                if morph2.pos1 == '接尾':
                    if not is_concat:
                        morph_modified, is_concat = self._concat_suffix(
                            morph1=morph1, morph2=morph2,
                            target_pos=target_pos, target_regex=target_regex,
                            suffix_pos2=suffix_pos2
                        )
                    else:
                        modified_morphs_list.pop(-1)
                        morph_modified, is_concat = self._concat_suffix(
                            morph1=morph_modified, morph2=morph2,
                            target_pos=target_pos, target_regex=target_regex,
                            suffix_pos2=suffix_pos2
                        )
                    modified_morphs_list.append(morph_modified)
                else:
                    if not is_concat:
                        modified_morphs_list.append(morph1)
                    is_concat = False

            if not is_concat:
                modified_morphs_list.append(morph2)

            self.morphs_list = modified_morphs_list

    def concat_suffix_like_noun(self, target_suffixes: List[str], target_word_len_min: int = 2):
        if len(self.morphs_list) < 2:
            return
        else:
            modified_morphs_list = []
            is_concat = False
            morph_modified = self.morphs_list[0]
            for morph1, morph2 in pairwise(self.morphs_list):
                if morph2.lemma in target_suffixes and len(morph1.lemma) >= target_word_len_min:
                    if not is_concat:
                        morph_modified, is_concat = self._concat_suffix(
                            morph1=morph1, morph2=morph2, target_pos='名詞'
                        )
                    else:
                        modified_morphs_list.pop(-1)
                        morph_modified, is_concat = self._concat_suffix(
                            morph1=morph_modified, morph2=morph2, target_pos='名詞'
                        )
                    modified_morphs_list.append(morph_modified)
                else:
                    if not is_concat:
                        modified_morphs_list.append(morph1)
                    is_concat = False

            if not is_concat:
                modified_morphs_list.append(morph2)

            self.morphs_list = modified_morphs_list

    @staticmethod
    def _concat_suffix(morph1: MeCabMorph, morph2: MeCabMorph, target_pos: Optional[str] = None,
                       target_regex: Optional[str] = None, suffix_pos2: Optional[Union[str, List[str]]] = None):
        morph_modified = morph1
        if morph_modified.pos != target_pos:
            return morph1, False
        else:
            if not target_regex:
                if not suffix_pos2:
                    morph_modified.surface += morph2.surface
                    morph_modified.lemma += morph2.lemma
                    morph_modified.pronounce += morph2.pronounce
                    return morph_modified, True
                else:
                    if morph2.pos2 == suffix_pos2:
                        morph_modified.surface += morph2.surface
                        morph_modified.lemma += morph2.lemma
                        morph_modified.pronounce += morph2.pronounce
                        return morph_modified, True
                    else:
                        return morph1, False
            else:
                if re.fullmatch(target_regex, morph_modified.surface):
                    morph_modified.surface += morph2.surface
                    morph_modified.lemma += morph2.lemma
                    morph_modified.pronounce += morph2.pronounce
                    return morph_modified, True
                else:
                    return morph1, False

    def concat_prefix(self, target_pos: Optional[str] = None, target_regex: Optional[str] = None,
                      prefix_pos1: Optional[Union[str, List[str]]] = None):
        if '接頭詞' not in [morph.pos for morph in self.morphs_list] or len(self.morphs_list) < 2:
            return
        else:
            modified_morphs_list = []
            is_concat = False
            for morph1, morph2 in pairwise(reversed(self.morphs_list)):
                if not is_concat:
                    if morph2.pos == '接頭詞':
                        morph_modified, is_concat = self._concat_prefix(
                            morph1=morph1, morph2=morph2,
                            target_pos=target_pos, target_regex=target_regex,
                            prefix_pos1=prefix_pos1
                        )
                        modified_morphs_list.append(morph_modified)
                    else:
                        modified_morphs_list.append(morph1)
                else:
                    is_concat = False
            if morph2.pos != '接頭詞' or not is_concat:
                modified_morphs_list.append(morph2)

            self.morphs_list = list(reversed(modified_morphs_list))

    @staticmethod
    def _concat_prefix(morph1: MeCabMorph, morph2: MeCabMorph, target_pos: Optional[str] = None,
                       target_regex: Optional[str] = None, prefix_pos1: Optional[Union[str, List[str]]] = None):
        morph_modified = morph1
        if morph_modified.pos != target_pos:
            return morph1, False
        else:
            if not target_regex:
                if not prefix_pos1:
                    morph_modified.surface = morph2.surface + morph_modified.surface
                    morph_modified.lemma = morph2.lemma + morph_modified.lemma
                    morph_modified.pronounce = morph2.pronounce + morph_modified.pronounce
                    return morph_modified, True
                else:
                    if morph2.pos1 == prefix_pos1:
                        morph_modified.surface = morph2.surface + morph_modified.surface
                        morph_modified.lemma = morph2.lemma + morph_modified.lemma
                        morph_modified.pronounce = morph2.pronounce + morph_modified.pronounce
                        return morph_modified, True
                    else:
                        return morph1, False
            else:
                if re.fullmatch(target_regex, morph_modified.surface):
                    morph_modified.surface = morph2.surface + morph_modified.surface
                    morph_modified.lemma = morph2.lemma + morph_modified.lemma
                    morph_modified.pronounce = morph2.pronounce + morph_modified.pronounce
                    return morph_modified, True
                else:
                    return morph1, False

    def concat_nouns(self, max_concats: int = 4, concat_nouns_type: List[str] = None):
        if len(self.morphs_list) >= 2:

            i = 0
            while i < max_concats:
                modified_morphs_list = []
                concat_flg = False
                for morph1, morph2 in pairwise(self.morphs_list):
                    if concat_flg:
                        concat_flg = False
                        continue

                    morph_modified = copy(morph1)
                    if morph1.pos == '名詞' and morph2.pos == '名詞':
                        if concat_nouns_type is None or \
                                (morph1.pos1 in concat_nouns_type and morph2.pos1 in concat_nouns_type):
                            morph_modified.surface += morph2.surface
                            morph_modified.lemma += morph2.lemma
                            concat_flg = True

                    modified_morphs_list.append(morph_modified)
                modified_morphs_list.append(morph2)

                if self.morphs_list == modified_morphs_list:
                    break
                else:
                    self.morphs_list = modified_morphs_list
                    i += 1

    def concat_katakana_nouns(self, pattern: regex.Pattern):
        if len(self.morphs_list) >= 2:

            modified_morphs_list = []
            concat_flg = False
            for morph1, morph2 in pairwise(self.morphs_list):
                if concat_flg:
                    concat_flg = False
                    continue

                morph_modified = copy(morph1)
                if morph1.pos == '名詞' and pattern.fullmatch(morph1.surface) and \
                        morph2.pos == '名詞' and pattern.fullmatch(morph2.surface):
                    morph_modified.surface += morph2.surface
                    morph_modified.lemma += morph2.lemma
                    concat_flg = True

                modified_morphs_list.append(morph_modified)
            modified_morphs_list.append(morph2)

            self.morphs_list = modified_morphs_list

    def remove_stopwords(self, stopwords: List[str]):
        self.morphs_list = [morph for morph in self.morphs_list if morph.lemma not in stopwords]

