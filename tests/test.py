import unittest
from takuge_toolkit.tokenizer.base import BaseMorph, BaseMorphList
from takuge_toolkit.tokenizer.mecab import MeCabMorph, MeCabMorphList, MeCabTokenizer


class TestBaseMorph(unittest.TestCase):
    def setUp(self) -> None:
        self.morph = BaseMorph(**dict(surface='ん', lemma='ん', pos='名詞', pos1='非自立', pos2='一般'))
        self.pos_combi = ('名詞', '非自立')
        self.pos_combi_list = [('名詞', '代名詞'), ('名詞', '非自立'), ('名詞', '接尾')]
        self.pos_list = ['名詞', '動詞', '形容詞']

    def test__is_pos_type_to_filter(self):
        self.assertTrue(self.morph._is_pos_type_to_filter(pos_combi=self.pos_combi))

    def test_is_pos_type_to_extract_exclude(self):
        self.assertFalse(self.morph.is_pos_type_to_extract(exclude_pos_combi_list=self.pos_combi_list))

    def test_is_pos_type_to_extract_extract(self):
        self.assertTrue(self.morph.is_pos_type_to_extract(extract_pos_combi_list=self.pos_list))

    def test_is_pos_type_to_extract_extract_and_exclude(self):
        self.assertFalse(self.morph.is_pos_type_to_extract(extract_pos_combi_list=self.pos_list,
                                                           exclude_pos_combi_list=self.pos_combi_list))


class TestBaseMorphList(unittest.TestCase):
    def setUp(self) -> None:
        self.morphs = BaseMorphList(morphs_list=[
            BaseMorph(surface='レビュー', lemma='レビュー', pos='名詞', pos1='サ変接続', pos2=None),
            BaseMorph(surface='読む', lemma='読む', pos='動詞', pos1='自立', pos2=None),
            BaseMorph(surface='と', lemma='と', pos='助詞', pos1='接続助詞', pos2=None),
            BaseMorph(surface='薄いっ', lemma='薄い', pos='形容詞', pos1='自立', pos2=None),
            BaseMorph(surface='ていう', lemma='ていう', pos='助詞', pos1='格助詞', pos2='連語'),
            BaseMorph(surface='声', lemma='声', pos='名詞', pos1='一般', pos2=None),
            BaseMorph(surface='も', lemma='も', pos='助詞', pos1='係助詞', pos2=None),
            BaseMorph(surface='結構', lemma='結構', pos='副詞', pos1='一般', pos2=None),
            BaseMorph(surface='ある', lemma='ある', pos='動詞', pos1='自立', pos2=None),
            BaseMorph(surface='ん', lemma='ん', pos='名詞', pos1='非自立', pos2='一般'),
            BaseMorph(surface='だ', lemma='だ', pos='助動詞', pos1=None, pos2=None),
            BaseMorph(surface='けど', lemma='けど', pos='助詞', pos1='接続助詞', pos2=None),
        ])
        self.exclude_pos_combi_list = [('名詞', '代名詞'), ('名詞', '非自立'), ('名詞', '接尾')]
        self.extract_pos_list = ['名詞', '動詞', '形容詞']

    def test_get_elements(self):
        self.assertListEqual(self.morphs.get_elements(element_type='lemma'),
                             ['レビュー', '読む', 'と', '薄い', 'ていう', '声', 'も', '結構', 'ある', 'ん', 'だ', 'けど'])

    def test_get_concatenated_elements(self):
        self.assertListEqual(self.morphs.get_concatenated_elements(element_types=['lemma', 'pos']),
                             ['レビュー_名詞', '読む_動詞', 'と_助詞', '薄い_形容詞', 'ていう_助詞', '声_名詞',
                              'も_助詞', '結構_副詞', 'ある_動詞', 'ん_名詞', 'だ_助動詞', 'けど_助詞'])

    def test_filter_morphs_by_pos(self):
        self.assertListEqual(
            self.morphs.filter_morphs_by_pos(extract_pos_combi_list=self.extract_pos_list,
                                             exclude_pos_combi_list=self.exclude_pos_combi_list).morphs_list,
            [BaseMorph(surface='レビュー', lemma='レビュー', pos='名詞', pos1='サ変接続', pos2=None),
             BaseMorph(surface='読む', lemma='読む', pos='動詞', pos1='自立', pos2=None),
             BaseMorph(surface='薄いっ', lemma='薄い', pos='形容詞', pos1='自立', pos2=None),
             BaseMorph(surface='声', lemma='声', pos='名詞', pos1='一般', pos2=None),
             BaseMorph(surface='ある', lemma='ある', pos='動詞', pos1='自立', pos2=None)])


if __name__ == '__main__':
    unittest.main()
