# -*- coding: utf-8 -*-
from typing import Union, Sequence, List, Tuple, Optional
from itertools import compress
from pprint import pformat
from dataclasses import dataclass
from abc import ABCMeta


@dataclass
class BaseMorph(metaclass=ABCMeta):
    """
    形態素解析の結果を属性として格納するdataclass
    """
    surface: str
    """表層形"""

    lemma: str
    """原形"""

    pos: str
    """品詞"""

    pos1: str
    """品詞細分類1"""

    pos2: str
    """品詞細分類2"""

    def __new__(cls, *args, **kwargs):
        dataclass(cls)
        return super().__new__(cls)

    def is_pos_type_to_extract(self,
                               extract_pos_combi_list: List[Union[Tuple[str, str], str]] = None,
                               exclude_pos_combi_list: List[Union[Tuple[str, str], str]] = None) -> bool:
        """除外したい品詞等・抽出したい品詞等に当てはまるかどうか？を判定する

        Args:
            extract_pos_combi_list: 抽出したい品詞のlist(例: ['名詞', '動詞', '形容詞']) or 品詞と品詞分類のtupleのlist(例: [('名詞', '代名詞'), ('名詞', '非自立'), ('名詞', '接尾')])
            exclude_pos_combi_list: 除外したい品詞のlist or 品詞と品詞分類のtupleのlist

        Returns:
            bool: 除外したい品詞等・抽出したい品詞等に当てはまるかどうか？

        """
        result_extract = True  # `extract_pos_combi_list`がない場合は、基本的に全部extractする想定
        result_exclude = False  # `exclude_pos_combi_list`がない場合は、基本的に全部excludeしない想定
        if extract_pos_combi_list:
            result_extract = any([self._is_pos_type_to_filter(pos_combi=extract_pos_combi)
                                  for extract_pos_combi in extract_pos_combi_list])
        if exclude_pos_combi_list:
            result_exclude = any([self._is_pos_type_to_filter(pos_combi=exclude_pos_combi)
                                  for exclude_pos_combi in exclude_pos_combi_list])
        return False if result_exclude else result_extract

    def _is_pos_type_to_filter(self, pos_combi: Union[Tuple[str, str], str]) -> bool:
        """フィルタリングしたい品詞等に当てはまるかどうか？を判定する

        Args:
            pos_combi: 品詞のlist(例: ['名詞', '動詞', '形容詞']) or 品詞と品詞分類のtupleのlist(例: [('名詞', '代名詞'), ('名詞', '非自立'), ('名詞', '接尾')])

        Returns:
            bool: フィルタリングしたい品詞等に当てはまるかどうか？

        """
        pos, pos1, pos2 = set_pos_combi(pos_combi)

        if pos2 is not None:
            result = self.pos == pos and self.pos1 == pos1 and self.pos2 == pos2
        elif pos1 is not None:
            result = self.pos == pos and self.pos1 == pos1
        else:
            result = self.pos == pos

        return result


class BaseMorphList:
    def __init__(self, morphs_list: Sequence[BaseMorph]):
        self.morphs_list = morphs_list

    def __repr__(self):
        return str(pformat(self.morphs_list))

    def __getitem__(self, index):
        return self.morphs_list[index]

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

    def filter_morphs_by_pos(self,
                             extract_pos_combi_list: List[Union[Tuple[str, str], str]] = None,
                             exclude_pos_combi_list: List[Union[Tuple[str, str], str]] = None):
        is_extract = [morph.is_pos_type_to_extract(extract_pos_combi_list, exclude_pos_combi_list)
                      for morph in self.morphs_list]
        self.morphs_list = list(compress(self.morphs_list, is_extract))
        return self


def set_pos_combi(pos_combi: Union[str, Union[Tuple[str], Tuple[str, str], Tuple[str, str, str]]]
                  ) -> Tuple[str, Optional[str], Optional[str]]:
    """品詞・品詞細分類の引数を展開する

    Args:
        pos_combi: 品詞のstr(例: '名詞') or 品詞と品詞分類のtuple(例: ('名詞', '代名詞', '一般'))

    Returns:
        Tuple[str, str, str]: 品詞, 品詞細分類1, 品詞細分類2のtuple

    Examples:
        >>> set_pos_combi(pos_combi='名詞')
        ('名詞', None, None)
        >>> set_pos_combi(pos_combi=('名詞', '代名詞'))
        ('名詞', '代名詞', None)
        >>> set_pos_combi(pos_combi=('名詞', '代名詞', '一般'))
        ('名詞', '代名詞', '一般)

    """
    pos, pos1, pos2 = None, None, None

    if type(pos_combi) == str:
        pos = pos_combi
    else:
        if len(pos_combi) >= 1:
            pos = pos_combi[0]
        if len(pos_combi) >= 2:
            pos1 = pos_combi[1]
        if len(pos_combi) >= 3:
            pos2 = pos_combi[2]

    return pos, pos1, pos2


if __name__ == '__main__':
    import doctest
    doctest.testmod()
