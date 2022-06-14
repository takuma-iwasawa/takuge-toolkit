# -*- coding: utf-8 -*-
from .base import Morph, MorphList
from typing import List, Dict, Iterable, Union, Optional
import MeCab
import re
import unicodedata
import regex
from more_itertools import pairwise
from copy import copy


class MeCabTokenizer:
    def __init__(self, tagger_option: Optional[str] = None, normalize_form='NFKC', normalize_digit=False,
                 stopwords: Optional[List[str]] = None):
        self.tagger = MeCab.Tagger(tagger_option) if tagger_option else MeCab.Tagger()
        self.normalize_form = normalize_form
        self.normalize_digit = normalize_digit
        self.stopwords = stopwords
        self.katakana_pattern = regex.compile(r'[\p{Script=Katakana}ー]+')

    def tokenize(self, sentence, extract_element_type: Union[str, List[str]] = 'lemma',
                 extract_pos=None, extract_pos1=None, exclude_pos1=None, exclude_pos2=None, remove_eos=True,
                 concat_suffix_target: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
                 concat_suffix_like_noun_target: Optional[List[str]] = None,
                 concat_prefix_target: Optional[Union[Dict[str, str], List[Dict[str, str]]]] = None,
                 concat_pos_targets: Optional[Dict[str, List[str]]] = None,
                 concat_katakana_nouns=False) -> List[str]:
        parsed_node_list = self.tokenize_as_mecab_format(sentence=sentence, remove_eos=remove_eos)
        return self.extract_words_from_mecab_format(
            parsed_node_list=parsed_node_list, extract_element_type=extract_element_type, extract_pos=extract_pos,
            extract_pos1=extract_pos1, exclude_pos1=exclude_pos1, exclude_pos2=exclude_pos2,
            concat_suffix_target=concat_suffix_target, concat_suffix_like_noun_target=concat_suffix_like_noun_target,
            concat_prefix_target=concat_prefix_target, concat_pos_targets=concat_pos_targets,
            concat_katakana_nouns=concat_katakana_nouns,
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
                                        concat_pos_targets: Optional[Dict[str, List[str]]] = None,
                                        concat_katakana_nouns=False) -> List[str]:
        morphs = MeCabMorphList([MeCabMorph(node) for node in parsed_node_list])

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

        if concat_katakana_nouns:
            morphs.concat_katakana_nouns(pattern=self.katakana_pattern)

        if self.stopwords:
            morphs.remove_stopwords(self.stopwords)

        morphs.filter_morphs_by_pos(extract_pos=extract_pos, extract_pos1=extract_pos1,
                                    exclude_pos1=exclude_pos1, exclude_pos2=exclude_pos2)

        # if concat_pos_targets:
        #     morphs.concat_pos(targets=concat_pos_targets)

        return morphs.get_concatenated_elements(element_types=extract_element_type)


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
        self.pos1 = morphemes[2] if morphemes[2] != '*' else None
        self.pos2 = morphemes[3] if morphemes[3] != '*' else None
        self.pos3 = morphemes[4] if morphemes[4] != '*' else None
        self.ctype = morphemes[5] if morphemes[5] != '*' else None
        self.cform = morphemes[6] if morphemes[6] != '*' else None
        try:
            self.pronounce = morphemes[8] if morphemes[8] != '*' else None
        except IndexError:
            self.pronounce = self.lemma


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

            # if morph2.pos1 != '接尾' or not is_concat:
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

    def concat_katakana_nouns(self, pattern: regex.Pattern):
        # pattern = regex.compile(r'[\p{Script=Katakana}ー]+')
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

    # def concat_pos(self, targets: Dict[str, List[str]]):
    #     if len(self.morphs_list) < 2:
    #         return
    #     else:
    #         sep = '、'
    #         for morph1, morph2 in pairwise(self.morphs_list):
    #             for target_pos, concat_pos in targets.items():
    #                 if morph1.pos == target_pos and morph2.pos in concat_pos:
    #                     morph_modified = copy(morph1)
    #                     morph_modified.surface += sep
    #                     morph_modified.lemma += sep
    #                     morph_modified.pronounce += sep
    #                     morph_modified, _ = self._concat_suffix(
    #                         morph1=morph_modified, morph2=morph2, target_pos=target_pos
    #                     )
    #                     self.morphs_list.append(morph_modified)

