# -*- coding: utf-8 -*-
from abc import ABC
from typing import Union, Optional, List, Iterable, Dict
import pandas as pd
from dask.diagnostics import ProgressBar
import numpy as np
from sklearn.base import ClassifierMixin, TransformerMixin
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split, BaseCrossValidator
from imblearn.under_sampling import RandomUnderSampler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, log_loss, accuracy_score
import mlflow
import mlflow.sklearn
import seaborn as sns
import japanize_matplotlib
from collections import OrderedDict
from time import time
from .pipeline_components import DropOthersLabelBinarizer, PredictProbaTransformer, MaxProbaLabelingWithOtherEstimator


ProgressBar().register()
sns.set()
japanize_matplotlib.japanize()


class ContainOthersClassifier:
    TOKEN_PATTERN = '(?u)\\b\\w+\\b'

    def __init__(self, model: ClassifierMixin, vectorizer: TransformerMixin, others_labels: Union[str, List[str]],
                 mlflow_experiment_name: Optional[str] = None, mlflow_add_params: Optional[Dict] = None):
        self.model = model
        self.vectorizer = vectorizer
        self.vectorizer.token_pattern = self.TOKEN_PATTERN
        self.others_labels = others_labels
        self.binarizer = DropOthersLabelBinarizer(other_label=others_labels)

        self.experiment_name = mlflow_experiment_name
        self.experiment_name_label = f'{self.experiment_name}_label' if self.experiment_name is not None else None
        self.mlflow_add_params = mlflow_add_params

        self.classes_ = OrderedDict()
        self.Y = None
        self.pipeline = self.create_pipeline()

    def create_pipeline(self):
        pipeline = Pipeline(steps=[
            ('vectorize', self.vectorizer),
            ('model', PredictProbaTransformer(MultiOutputClassifier(estimator=self.model, n_jobs=-1))),
            ('estimate', MaxProbaLabelingWithOtherEstimator())
        ], verbose=True)
        return pipeline

    def fit(self, X, y):
        self.Y = self.binarizer.fit_transform(y)
        self.pipeline.fit(X, self.Y)

        for i, cls in enumerate(self.binarizer.classes_):
            self.classes_[i] = cls

        return self

    def predict(self, X):
        print('predict...')
        t = time()
        pred = self.pipeline.predict(X)
        print(f'time: {time() - t}')
        return np.vectorize(lambda x: self.classes_[x])(pred)

    def predict_proba(self, X):
        print('predict proba...')
        t = time()
        proba = self.pipeline.predict_proba(X)
        print(f'time: {time() - t}')
        return proba

    def report(self, y_true, y_pred, y_pred_proba):
        df_report = pd.DataFrame(classification_report(y_pred=y_pred, y_true=y_true, output_dict=True)).T
        df_report['accuracy'] = accuracy_score(y_true=y_true, y_pred=y_pred)
        df_report['log_loss'] = log_loss(y_true=y_true, y_pred=y_pred_proba)
        return df_report

    def fit_predict_stratify(self, X, y, stratify: pd.DataFrame, test_size=0.3, seed=42):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=seed, stratify=stratify)
        self.fit(X_train, y_train)
        y_pred = self.predict(X_test)
        return X_test, y_test, y_pred

    def stratified_cv(self, inputs, stratify_cols: Union[str, List[str]],
                      folds=5, seed=42) -> pd.DataFrame:
        t_all = time()
        splits = self._get_stratified_cv_splits(features=inputs, stratify_cols=stratify_cols, folds=folds, seed=seed)
        reports = []
        for i, (train_idx, test_idx) in enumerate(splits):
            print(f'current CV: {i + 1} / {folds}')
            X_train, X_test = inputs.loc[train_idx, 'corpus'], inputs.loc[test_idx, 'corpus']
            y_train, y_test = inputs.loc[train_idx, 'y'], inputs.loc[test_idx, 'y']

            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)
            y_pred_proba = self.predict_proba(X_test)
            df_report = self.report(y_true=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba)
            reports.append(df_report)
        df_reports = (
            pd.concat(reports, axis=0).reset_index()
              .groupby('index').agg(['mean', 'std'])
        )

        log_to_mlflow = self.experiment_name is not None
        df_reports_return = self.log_reports_to_mlflow(df_report=df_reports) if log_to_mlflow else df_reports
        print(f'\nwhole time: {round((time() - t_all) / 60, 1)}min')
        return df_reports_return

    def _get_stratified_cv_splits(self, features, stratify_cols: Union[str, List[str]],
                                  folds: int, seed: int) -> Iterable:
        stratified_col = self._get_stratify_cols(features=features, stratify_cols=stratify_cols)
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
        splits = skf.split(X=features, y=stratified_col)
        return splits

    @staticmethod
    def _get_stratify_cols(features: pd.DataFrame, stratify_cols: Union[str, List[str]]) -> pd.Series:
        if type(stratify_cols) == str:
            stratified_col = features[stratify_cols]
        else:
            stratified_col = \
                features[stratify_cols[0]].astype(str).str.cat(features[stratify_cols[1:]].astype(str), sep=' + ')
        return stratified_col


class MLFlowUtils:
    def log_reports_to_mlflow(self, experiment_name, pipeline, df_report):
        df_report = (
            df_report.drop(index=['accuracy', 'macro avg'], columns='support')
                     .drop(columns='std', level=1)
                     .droplevel(level=1, axis=1)
        )

        # 全体のmetricsを記録
        print('log average params & metrics to mlflow...')
        total_index = 'weighted avg'
        df_report_total = df_report.loc[total_index, :]
        try:
            mlflow.create_experiment(experiment_name)
        except:
            mlflow.set_experiment(experiment_name)
        with mlflow.start_run(nested=True):
            self.log_params_to_mlflow()
            mlflow.sklearn.log_model(pipeline, 'pipeline')
            for metric in df_report_total.index:
                mlflow.log_metric(metric, df_report_total[metric])

        print('done.')
        return df_report

    def log_params_to_mlflow(self, label: Optional[str] = None):
        mlflow.log_param('model', self.model.__class__.__name__)
        mlflow.log_param('vectorizer', self.vectorizer.__class__.__name__)
        if self.mlflow_add_params is not None:
            for param, value in self.mlflow_add_params.items():
                mlflow.log_param(param, value)
        if label is not None:
            mlflow.log_param('label', label)

    @staticmethod
    def load_model_from_mlflow(run_id, model_name):
        return mlflow.sklearn.load_model(f'runs:/{run_id}/{model_name}/')


class CompareModels:
    def __init__(self, mlflow_experiment_name, pipelines: Dict[str, Pipeline], threshold=0.5):
        self.mlflow_experiment_name = mlflow_experiment_name
        self.pipelines = pipelines
        self.threshold = threshold

    def load_pipelines(self, input_dict: Dict):
        assert input_dict.keys() == self.pipelines.keys()

        for name, pipeline in self.pipelines.items():
            pipeline.load_model_from_mlflow(label_run_id_dict=input_dict[name])

    def predict(self, inputs: pd.DataFrame):
        dict_proba = dict()
        for name, pipeline in self.pipelines.items():
            print(f'predict: {name} ...')
            dict_proba[name] = pipeline.predict(inputs, threshold=self.threshold)

        df_proba = (
            pd.concat(dict_proba, axis=1)
              .reorder_levels([1, 0], axis=1)
              .sort_index(level=0, axis=1)
        )
        return df_proba


class UnderBaggingKFold(BaseCrossValidator, ABC):
    """CV に使うだけで UnderBagging できる KFold 実装

    NOTE: 少ないクラスのデータは各 Fold で重複して選択される"""

    def __init__(self, n_splits=5, shuffle=True, random_state=None, minority_ratio=0.5,
                 test_size=0.3, whole_testing=False):
        """
        :param n_splits: Fold の分割数
        :param shuffle: 分割時にデータをシャッフルするか
        :param random_state: 乱数シード
        :param minority_ratio: 割合が少ない方のカテゴリを、多い方のカテゴリとの比でどれくらいにするか
        :param test_size: Under-sampling された中でテスト用データとして使う割合
        :param whole_testing: Under-sampling で選ばれなかった全てのデータをテスト用データに追加するか
        """
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        self.test_size = test_size
        self.whole_testing = whole_testing
        self.minority_ratio = minority_ratio

        if random_state is not None:
            np.random.seed(seed=random_state)
        self.random_states = np.random.randint(0, 100, n_splits)

        # 分割数だけ Under-sampling 用のインスタンスを作っておく
        self.samplers_ = [
            RandomUnderSampler(sampling_strategy=minority_ratio, random_state=random_state)
            for random_state in self.random_states
        ]

    def split(self, X, y=None, groups=None):
        """データを学習用とテスト用に分割する"""
        if X.ndim < 2:
            # RandomUnderSampler#fit_resample() は X が 1d-array だと文句を言う
            X = np.vstack(X)

        for i in range(self.n_splits):
            # データを Under-sampling して均衡データにする
            sampler = self.samplers_[i]
            _, y_sampled = sampler.fit_resample(X, y)
            # 選ばれたデータのインデックスを取り出す
            sampled_indices = sampler.sample_indices_

            # 選ばれたデータを学習用とテスト用に分割する
            split_data = train_test_split(sampled_indices,
                                          shuffle=self.shuffle,
                                          test_size=self.test_size,
                                          stratify=y_sampled,
                                          random_state=self.random_state,
                                          )
            train_indices, test_indices = split_data

            if self.whole_testing:
                # Under-sampling で選ばれなかったデータをテスト用に追加する
                mask = np.ones(len(X), dtype=np.bool)
                mask[sampled_indices] = False
                X_indices = np.arange(len(X))
                non_sampled_indices = X_indices[mask]
                test_indices = np.concatenate([test_indices,
                                               non_sampled_indices])

            yield train_indices, test_indices

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
