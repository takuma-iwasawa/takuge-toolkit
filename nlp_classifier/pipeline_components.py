# -*- coding: utf-8 -*-
from typing import Union, Optional, List, Tuple
import re
import numpy as np
import pandas as pd
import swifter
import dask.dataframe as dd
from functools import partial, reduce
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.multioutput import MultiOutputClassifier
from JapaneseTokenizer import MecabWrapper, JumanppWrapper
from warnings import warn


class TokenizeTransformer(TransformerMixin):
    """
    テキストカラムを形態素解析する
    """
    CONTAINS_NULL_MESSAGE = "Text column contains null."

    def __init__(self, tokenizer: Union[MecabWrapper, JumanppWrapper]):
        """
        :param tokenizer: 形態素解析器
        """
        self.tokenizer = tokenizer
        self.is_dask = None

    def fit(self, X: Union[pd.Series, dd.Series], y=None):
        self.is_dask = isinstance(X, dd.Series)

        null_cnt = X.isnull().compute().sum() if self.is_dask else X.isnull().sum()
        if null_cnt > 0:
            warn(self.CONTAINS_NULL_MESSAGE)

        return self

    def transform(self, X: Union[pd.Series, dd.Series],
                  pos_condition: Optional[List[Tuple[str]]] = None, normalize_type='NFKC', normalize_number=False):
        sr_corpus = self._create_corpus(sr_text=X, normalize_number=normalize_number,
                                        normalize_type=normalize_type, pos_condition=pos_condition)
        if self.is_dask:
            sr_corpus = sr_corpus.compute()
        return sr_corpus

    def _create_corpus(self, sr_text: Union[pd.Series, dd.Series], normalize_type: Optional[str], normalize_number: bool,
                       pos_condition: Optional[List[Tuple[str]]] = None) -> Union[pd.Series, dd.Series]:
        """
        テキストカラムを形態素解析する

        :param sr_text: 形態素解析したいテキストカラム
        :param normalize_type: 正規化する形式(したくない場合はNone)
        :param normalize_number: 数字を正規化(=全て`0`にする)するかどうか
        :param pos_condition: 品詞を絞る場合、そのリスト(したくない場合はNone)
        :return: 形態素解析したテキストカラム
        """
        if normalize_type is not None:
            sr_text = sr_text.str.normalize(normalize_type)

        tokenize_func = partial(self._tokenize_sentence, normalize_number=normalize_number, pos_condition=pos_condition)
        if self.is_dask:
            meta = sr_text.head(1).apply(tokenize_func)
            sr_corpus = sr_text.apply(tokenize_func, meta=meta)
        else:
            sr_corpus = sr_text.swifter.apply(tokenize_func)

        return sr_corpus

    def _tokenize_sentence(self, sentence: str, normalize_number: bool,
                           pos_condition: Optional[List[Tuple[str]]] = None) -> str:
        """
        文を形態素解析する

        :param sentence: 形態素解析したい文
        :param normalize_number: 数字を正規化(=全て`0`にする)するかどうか
        :param pos_condition: 品詞を絞る場合、そのリスト(したくない場合はNone)
        :return: 形態素解析し、スペースでつなげた文字列
        """
        if isinstance(sentence, str):
            sentence = re.sub('[0-9０-９]+', '0', sentence) if normalize_number else sentence

            if pos_condition is None:
                tokenized_sentence = ' '.join(map(str, self.tokenizer.tokenize(sentence=sentence, return_list=True)))
            else:
                tokenized_sentence = ' '.join(map(str, self.tokenizer.tokenize(sentence=sentence, return_list=False)
                                                  .filter(pos_condition=pos_condition).convert_list_object()))
        else:
            tokenized_sentence = ['']
        return tokenized_sentence


class DropOthersLabelBinarizer(LabelBinarizer, LabelEncoder):
    """
    `その他`をdropした上でLabelBinarizerを施したものを返す
    (MultiOutputClassifierのfitの引数(Y)になる)
    """
    def __init__(self, other_label: Union[str, List[str]]):
        """
        :param other_label: `その他`に属するlabel(群)
        """
        super(DropOthersLabelBinarizer, self).__init__(self=self)
        self.other_label = other_label
        self.classes_ = None

    def fit(self, X, y=None):
        super(DropOthersLabelBinarizer, self).fit(X)
        return self

    def transform(self, X, y=None):
        Y = pd.DataFrame(super(DropOthersLabelBinarizer, self).transform(X), columns=self.classes_)

        if isinstance(self.other_label, str):
            _labels = list(filter(lambda x: self.other_label not in x, self.classes_)) + [self.other_label]
        else:
            _labels = [cls for cls in self.classes_
                       for others_label in self.other_label if cls != others_label] + self.other_label
        self.classes_ = _labels

        return Y.drop(columns=self.other_label)

    def fit_transform(self, X, y=None):
        return self.fit(X).transform(X)

    def get_feature_names(self):
        return self.classes_


class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Pipelineの中で、モデリングに使用するFeatureを選択して次の処理に託すために使用する
    """

    def __init__(self, col_name: Union[str, List[str]]):
        """
        :param col_name: SelectするColumn名のリスト
        """
        self.col_name = col_name

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X[self.col_name]


class PredictProbaTransformer(BaseEstimator, TransformerMixin):
    """
    Transformerとしてpredict_probaを行う
    アルゴリズムをMultiOutputClassifierに食わせた後で、各labelに属するprobaを取り出す時とかで使う
    (MultiOutputClassifierだと、各labelにつき属する/属しない確率の2columnが返ってきてしまうので)
    """

    def __init__(self, clf):
        """
        :param clf: モデリングに使用するアルゴリズム
        """
        self.clf = clf

    def fit(self, X, y):
        if self.clf is not None:
            self.clf.fit(X, y)
        return self

    def transform(self, X):
        is_multi_output = isinstance(self.clf, MultiOutputClassifier)
        if self.clf is not None:
            # Drop the 2nd column but keep 2d shape
            # because FeatureUnion wants that
            if is_multi_output:
                probas_tmp = self.clf.predict_proba(X)
                proba = reduce(
                    lambda x, y: np.hstack([x, y]),
                    [proba[:, [1]] for proba in probas_tmp]
                )
            else:
                proba = self.clf.predict_proba(X)[:, [1]]
            return proba

        return X

    # This method is important for correct working of pipeline
    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


class MaxProbaLabelingWithOtherEstimator(BaseEstimator):
    """
    その他の(どのラベルにも属さない)確率を算出する
    """

    def fit(self, X, y=None):
        """
        :param X: 各columnごとに該当するlabelのprobaが入ったmatrix
        :param y: 無くてOK
        """
        return self

    def predict(self, X):
        """
        その他(どのラベルにも属さない)を含め、最も確率が高いlabelを返す

        :param X: 各columnごとに該当するlabelのprobaが入ったmatrix
        :return:
        """
        return np.argmax(self.predict_proba(X=X), axis=1)

    def predict_proba(self, X):
        """
        その他の(どのラベルにも属さない)確率を算出し、カラムとして追加して返す

        :param X: 各columnごとに該当するlabelのprobaが入ったmatrix
        :return: Xに`その他`のprobaのcolumnが付与されたmatrix
        """
        other_proba = 1 - X.max(axis=1)
        return np.insert(X, X.shape[1], other_proba, axis=1)


