# -*- coding: utf-8 -*-
from typing import Union, Iterable, List, Optional


class Morph:
    def __init__(self):
        self.surface = None
        self.lemma = None
        self.pos = None
        self.pos1 = None
        self.pos2 = None

    def __str__(self):
        return f'surface: {self.surface}\tlemma: {self.lemma}\tpos: {self.pos}\tpos1: {self.pos1}\tpos2: {self.pos2}'


class MorphList:
    def __init__(self, morphs_list: Iterable[Morph]):
        self.morphs_list = morphs_list

    def get_elements(self, element_type='lemma'):
        try:
            return [getattr(morph, element_type) for morph in self.morphs_list]
        except AttributeError:
            raise AttributeError("element_type should be in ['surface', 'lemma', 'pos', 'pos1']")

    def get_concatenated_elements(self, element_types: Union[List[str], str]):
        if type(element_types) == str:
            return self.get_elements(element_type=element_types)
        else:
            return list(map(
                lambda elements: '_'.join([element for element in elements if element]),
                zip(*[self.get_elements(element_type=element_type) for element_type in element_types])
            ))

    def filter_morphs_by_pos(self, extract_pos: Optional[Union[List[str], str]] = None,
                             extract_pos1: Optional[Union[List[str], str]] = None,
                             exclude_pos1: Optional[Union[List[str], str]] = None,
                             exclude_pos2: Optional[Union[List[str], str]] = None):
        if type(extract_pos) == str:
            extract_pos = [extract_pos]
        if type(extract_pos1) == str:
            extract_pos1 = [extract_pos1]
        if type(exclude_pos1) == str:
            exclude_pos1 = [exclude_pos1]
        if type(exclude_pos2) == str:
            exclude_pos2 = [exclude_pos2]

        if any([extract_pos, extract_pos1, exclude_pos1, exclude_pos2]):
            modified_morphs = []
            if extract_pos:
                modified_morphs = [morph for morph in self.morphs_list if morph.pos in extract_pos]
            if extract_pos1:
                modified_morphs = [morph for morph in modified_morphs if morph.pos1 in extract_pos1]
            if exclude_pos1:
                modified_morphs = [morph for morph in modified_morphs if morph.pos1 not in exclude_pos1]
            if exclude_pos2:
                modified_morphs = [morph for morph in modified_morphs if morph.pos2 not in exclude_pos2]

            self.morphs_list = modified_morphs

            # if exclude_pos1:
            #     if extract_pos:
            #         if extract_pos1:
            #             self.morphs_list = [
            #                 morph for morph in self.morphs_list
            #                 if morph.pos in extract_pos and morph.pos1 in extract_pos1 and morph.pos1 not in exclude_pos1
            #             ]
            #         else:
            #             self.morphs_list = [
            #                 morph for morph in self.morphs_list
            #                 if morph.pos in extract_pos and morph.pos1 not in exclude_pos1
            #             ]
            #     else:
            #         if extract_pos1:
            #             self.morphs_list = [
            #                 morph for morph in self.morphs_list
            #                 if morph.pos1 in extract_pos1 and morph.pos1 not in exclude_pos1
            #             ]
            #         else:
            #             self.morphs_list = [
            #                 morph for morph in self.morphs_list
            #                 if morph.pos1 not in exclude_pos1
            #             ]
            # else:
            #     if extract_pos:
            #         if extract_pos1:
            #             self.morphs_list = [
            #                 morph for morph in self.morphs_list
            #                 if morph.pos in extract_pos and morph.pos1 in extract_pos1
            #             ]
            #         else:
            #             self.morphs_list = [
            #                 morph for morph in self.morphs_list
            #                 if morph.pos in extract_pos
            #             ]
            #     else:
            #         self.morphs_list = [
            #             morph for morph in self.morphs_list
            #             if morph.pos1 in extract_pos1
            #         ]

    # def filtering_condition(self, morph: Morph):


