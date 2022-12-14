# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
import re
from functools import reduce as ftreduce
from typing import Union, Tuple, List, Optional, Literal

import numpy as np
import pandas as pd
import scipy.sparse as sp
from gensim.models import Word2Vec
from negima import MorphemeMerger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, _document_frequency
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import Pipeline
from tqdm import tqdm

SEED = 42
tqdm.pandas()


class TokenizerTransformer(BaseEstimator, TransformerMixin):
    def __int__(self, tokenizer):
        self.tokenizer = tokenizer

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.Series, y=None, **kwargs):
        return X.swifter.apply(
            lambda sentence: self.tokenizer.tokenize(sentence=sentence, **kwargs),
            # meta=tuple
        )


class PhraseExtractorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, rule_csv_path: str, normalize_digit=False):
        self.rule_csv_path = rule_csv_path
        self.normalize_digit = normalize_digit
        self.mm = MorphemeMerger()
        self.pattern = re.compile(r'\d+')

    def fit(self, X, y=None):
        self.mm.set_rule_from_csv(self.rule_csv_path)
        return self

    def transform(self, X: pd.Series, y=None):
        return X.swifter.apply(
            lambda sentence:
                self.mm.get_rule_pattern(self.pattern.sub('0', str(sentence)))[0] if self.normalize_digit
                else self.mm.get_rule_pattern(sentence)[0],
        ).rename(f'{X.name}_phrase')


class VocabExtractorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, **kwargs):
        self.vectorizer = CountVectorizer(**kwargs)

    def fit(self, X: pd.Series, y=None):
        self.vectorizer.fit(X.swifter.apply(' '.join))
        return self

    def transform(self, X, y=None):
        return pd.Series(self.vectorizer.get_feature_names_out())


class FittedWord2VecTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, w2v: Union[str, Word2Vec], output_feature_name: str,
                 tokenizer, pooling: str = Literal['mean', 'max']):
        self.w2v = Word2Vec.load(w2v) if type(w2v) == str else w2v
        self.output_feature_name = output_feature_name
        self.tokenizer = tokenizer
        self.pooling = pooling

    def fit(self, X, y=None):
        return self

    def transform(self, X: Union[pd.Series, np.ndarray], y=None, **kwargs):
        return pd.Series(X.swifter.apply(lambda word: self._transform(word=word, **kwargs)).values,
                         index=X.values, name=self.output_feature_name).dropna()

    def _transform(self, word, **kwargs):
        tokens = self.tokenizer.tokenize(sentence=word, **kwargs)
        vectors = []
        for token in tokens:
            try:
                vectors.append(self.w2v.wv[token])
            except KeyError:
                continue
        try:
            return np.mean(vectors, axis=0) if self.pooling == 'mean' else np.max(vectors, axis=0)
        except:
            return None


class SimilarityMatrixTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: Tuple[pd.Series], y=None):
        df_similarity_matrix = pd.DataFrame(cosine_similarity(
            np.array(X[0].values.tolist()),
            np.array(X[1].values.tolist()),
        ), index=X[0].index, columns=X[1].index)

        df_similarity_matrix.index.name = X[0].name
        df_similarity_matrix.columns.name = X[1].name

        return df_similarity_matrix


class ExtractTopNSimilaritiesPerWordTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, topn_per_feature):
        self.topn_per_feature = topn_per_feature - 1

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame, y=None):
        similarity_matrix = X.values
        th = -np.partition(
            -similarity_matrix, kth=self.topn_per_feature,
            axis=1
        )[:, self.topn_per_feature]
        mat_args = np.argwhere(similarity_matrix >= th[:, None])
        result = pd.DataFrame(
            np.vstack([
                X.index[mat_args[:, 0]],
                X.columns[mat_args[:, 1]],
                similarity_matrix[similarity_matrix >= th[:, None]]
            ]).T,
            columns=['word', '最下位キーワード（漢字）', 'similarity']
        ).set_index(['word', '最下位キーワード（漢字）'])
        return result


class MeltVectorizerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vectorizer: Pipeline):
        self.vectorizer = vectorizer
        self.index = None
        self.id_dict = None
        self.vocabs = None

    def fit(self, X: pd.Series, y=None):
        self.index = X.index
        self.id_dict = dict(enumerate(X.index))
        self.vectorizer.fit(X.swifter.apply(' '.join))
        self.vocabs = self.vectorizer.named_steps['counter'].get_feature_names_out()
        return self

    def transform(self, X, y=None):
        vectors = self.vectorizer.transform(X.swifter.apply(' '.join))
        df_word_weights = (
            pd.DataFrame(np.concatenate([np.argwhere(vectors != 0), vectors[vectors != 0].T], axis=1),
                         columns=['index_col', 'word_id', 'weight'])
                .astype({'index_col': int, 'word_id': int})
                .pipe(lambda df: df.assign(index_col=df['index_col'].swifter.apply(lambda x: self.id_dict[x])))
                .pipe(lambda df: df.assign(word=df['word_id'].swifter.apply(lambda i: self.vocabs[i])))
                .drop(columns='word_id')
        )

        if type(self.index) == pd.MultiIndex:
            df_word_weights.index = pd.MultiIndex.from_tuples(df_word_weights['index_col'], names=self.index.names)
            df_word_weights.drop(columns='index_col', inplace=True)
        else:
            df_word_weights = df_word_weights.rename(columns={'index_col': self.index.name}).set_index(self.index.name)

        return df_word_weights.set_index('word', append=True).sort_index()


class FilterWordByWeightsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, topn: int, feature_name: str, by='similarity', adjust_weight_type: str = None):
        self.topn = topn
        self.by = by
        self.feature_name = feature_name
        self.adjust_weight_type = adjust_weight_type

    def fit(self, X: pd.Series, y=None):
        return self

    def transform(self, X, y=None):
        X['weight_adjusted'] = (
            X['weight'].apply(np.sqrt) if self.adjust_weight_type == 'sqrt'
            else X['weight'].apply(np.log) if self.adjust_weight_type == 'log'
            else X['weight']
        )
        X['weight_similarity'] = X['weight_adjusted'] * X['similarity']

        return (
            X.reset_index()
                .sort_values(['情報番号', self.by], ascending=[True, False])
                .drop_duplicates(subset=['情報番号', '最下位キーワード（漢字）'], keep='first')  # 同じ内容等KWがサジェストされた場合、スコアが最も高いもののみ残す
                .pipe(lambda df: df[df.groupby('情報番号')[self.by].rank(ascending=False) <= self.topn])
                .set_index('情報番号')
                .drop(columns='weight_adjusted')
        )


class PredictContentKeywordsTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name):
        self.feature_name = feature_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        try:
            return (
                X.groupby('情報番号').apply(lambda grouped: grouped.values)
                    .rename(f'KWs_with_{self.feature_name}')
                    .pipe(lambda sr: sr.to_frame().join(
                        sr.swifter.apply(lambda similarities: similarities[:, 1])
                            .swifter.apply(set)
                            .rename(f'pred_{self.feature_name}')
                    ))
            )
        except TypeError:
            return pd.Series([], name=f'pred_{self.feature_name}', dtype='object')


class ContentsKeywordsEstimator(BaseEstimator):
    def __init__(self, pred_col_prefix: str):
        self.pred_col_prefix = pred_col_prefix

    def fit(self, X: pd.DataFrame, y: pd.Series):
        return self

    def predict(self, X: pd.DataFrame):
        return X.assign(
            pred=X.filter(regex=f'^{self.pred_col_prefix}')
                .pipe(lambda df: df.swifter.apply(
                    lambda row: ftreduce(lambda x, y: x.union(y), filter(lambda x: pd.notnull(x), list(row))),
                    axis=1
                ))
        )

    def fit_predict(self, X: pd.DataFrame, y=None):
        return self.fit(X, y).predict(X)

    # def score(self, X: pd.DataFrame, y: pd.Series):
    #     pred = self.fit_predict(X, y)
    #     correct = (
    #         y.swifter.apply(set).to_frame()
    #             .join(pred['pred'], how='outer')
    #             .swifter.apply(
    #                 lambda row: set(row[y.name]) & row['pred']
    #                 if pd.notnull(row['pred']) else None,
    #             axis=1
    #         )
    #     )
    #     return correct.swifter.apply(len).sum() / correct.swifter.apply(len).sum()


class BM25Transformer(BaseEstimator, TransformerMixin):
    """
    Parameters
    ----------
    use_idf : boolean, optional (default=True)
    k1 : float, optional (default=2.0)
    b : float, optional (default=0.75)

    References
    ----------
    Okapi BM25: a non-binary model - Introduction to Information Retrieval
    http://nlp.stanford.edu/IR-book/html/htmledition/okapi-bm25-a-non-binary-model-1.html
    """
    def __init__(self, use_idf=True, k1=2.0, b=0.75):
        self.use_idf = use_idf
        self.k1 = k1
        self.b = b
        self._idf_diag = None

    def fit(self, X, y=None):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        """
        if not sp.issparse(X):
            X = sp.csc_matrix(X)
        if self.use_idf:
            n_samples, n_features = X.shape
            df = _document_frequency(X)
            idf = np.log((n_samples - df + 0.5) / (df + 0.5))
            self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)
        return self

    def transform(self, X, y=None, copy=True):
        """
        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            document-term matrix
        copy : boolean, optional (default=True)
        """
        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        # Document length (number of terms) in each row
        # Shape is (n_samples, 1)
        dl = X.sum(axis=1)
        # Number of non-zero elements in each row
        # Shape is (n_samples, )
        sz = X.indptr[1:] - X.indptr[0:-1]
        # In each row, repeat `dl` for `sz` times
        # Shape is (sum(sz), )
        # Example
        # -------
        # dl = [4, 5, 6]
        # sz = [1, 2, 3]
        # rep = [4, 5, 5, 6, 6, 6]
        rep = np.repeat(np.asarray(dl), sz)
        # Average document length
        # Scalar value
        avgdl = np.average(dl)
        # Compute BM25 score only for non-zero elements
        data = X.data * (self.k1 + 1) / (X.data + self.k1 * (1 - self.b + self.b * rep / avgdl))
        X = sp.csr_matrix((data, X.indices, X.indptr), shape=X.shape)

        if self.use_idf:
            # check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

            expected_n_features = self._idf_diag.shape[0]
            if n_features != expected_n_features:
                raise ValueError("Input has n_features=%d while the model"
                                 " has been trained with n_features=%d" % (
                                     n_features, expected_n_features))
            # *= doesn't work
            X = X * self._idf_diag

        return X
