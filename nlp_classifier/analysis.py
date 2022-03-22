# -*- coding: utf-8 -*-
from typing import Dict, Sequence, Iterable, Union
from pathlib import PosixPath
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from itertools import product, chain
from functools import reduce, partial
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.figure_factory as ff
import plotly.express as px
import pickle
from datetime import datetime, timedelta, timezone
import enum
from logging import getLogger

logger = getLogger(__name__)
JST = timezone(timedelta(hours=+9), 'JST')
DATETIME_STR = datetime.now(JST).strftime('%Y%m%d%H%S')


@enum.unique
class FileName(enum.Enum):
    EXCEL = f'{DATETIME_STR}_output.xlsx'
    Y_DIST = f'{DATETIME_STR}_y_distribution.png'
    SLEN_DIST = f'{DATETIME_STR}_strings_len_distribution.png'
    F1_COMPARE = f'{DATETIME_STR}_f1_compared_with_random.png'
    CONFUSION_MATRIX_MODEL = f'{DATETIME_STR}_confusion_matrix.png'
    CONFUSION_MATRIX_RANDOM = f'{DATETIME_STR}_confusion_matrix_random.png'
    WORD_WEIGHTS = f'{DATETIME_STR}_word_weights_ranking.png'
    PIPELINE = f'{DATETIME_STR}_pipeline_without_tokenizer.pkl'


@enum.unique
class SheetName(enum.Enum):
    Y_DIST = 'y_distribution'
    SLEN_DIST = 'text_len_dist'
    METRICS = 'metrics_report'
    SAMPLING = 'results_sampling'
    WORD_WEIGHTS_RANKING = 'word_weights_ranking'
    WORD_WEIGHTS_RESULT = 'word_weights_result_origin'
    MI = 'mi_ranking'


def output_results(pipeline_without_tokenizer: Pipeline, df_predicted: pd.DataFrame, X_tokenized: pd.Series,
                   text_col_name: str, label_col_name: str, y_random: Iterable, save_dir: Union[str, PosixPath],
                   fig_dir: Union[str, PosixPath], pipeline_dir: Union[str, PosixPath], seed: int):
    """
    アウトプットを出力する。出力するファイルリストはREADME参照
    https://gitlab.com/insight-tech-ds/common/classification-template/-/blob/master/README.md

    :param pipeline_without_tokenizer: 形態素解析器を除いたpipeline
    :param df_predicted: 予測結果付きのデータ
    :param X_tokenized: 形態素解析後のテキストカラム
    :param text_col_name: 元(形態素解析前)のテキストカラム名
    :param label_col_name: 目的変数のカラム名
    :param y_random: ランダム分類器による予測結果
    :param save_dir: 出力先ディレクトリのパス
    :param fig_dir: 画像ファイルの出力先パス
    :param pipeline_dir: パイプラインの出力先パス
    :param seed: 乱数のseed
    """
    y_true, y_pred = df_predicted[label_col_name], df_predicted['y_pred']

    # output作成
    fig_y_dist, df_y_dist = plot_label_dist(y=y_true)
    fig_slen_dist, sr_slen_dist = plot_text_len_distribution(text_col=df_predicted[text_col_name])
    fig_cm = plot_confusion_matrix(y_true=y_true, y_pred=y_pred)
    fig_cm_random = plot_confusion_matrix(y_true=y_true, y_pred=y_random)
    df_metrics = classification_report_compared_with_random(y_true=y_true, y_pred=y_pred, y_random=y_random)
    fig_f1_compare = plot_metric_compared_with_random(df_metrics=df_metrics, metric='f1-score')
    df_sampled = sampling_predict_results(df_predicted=df_predicted, label_col_name=label_col_name, seed=seed)
    df_word_weights_result, df_word_weights_ranking = calc_words_wights(pipeline=pipeline_without_tokenizer)
    fig_word_weights = plot_ranking(
        df_labels=df_word_weights_ranking['word'], df_values=df_word_weights_ranking['ratio_log_proba'].astype(float),
        rank_limit=50, title='単語の重みランキング', cmap='Reds_r'
    )
    df_mi = calc_mi_between_label_and_word(vectorizer=pipeline_without_tokenizer.named_steps['vectorizer'], X=X_tokenized, y=y_true)

    # グラフ画像ファイルを保存
    def _save_plotly_fig(fig: Union[go.Figure, go.FigureWidget], filename: FileName):
        fig.write_image(str(fig_dir.joinpath(filename.value)), engine="kaleido")
    _save_plotly_fig(fig_y_dist, FileName.Y_DIST)
    _save_plotly_fig(fig_slen_dist, FileName.SLEN_DIST)
    _save_plotly_fig(fig_cm, FileName.CONFUSION_MATRIX_MODEL)
    _save_plotly_fig(fig_cm_random, FileName.CONFUSION_MATRIX_RANDOM)
    _save_plotly_fig(fig_f1_compare, FileName.F1_COMPARE)
    _save_plotly_fig(fig_word_weights, FileName.WORD_WEIGHTS)

    # Excelシートを保存
    with pd.ExcelWriter(path=str(save_dir.joinpath(FileName.EXCEL.value))) as excel:
        df_y_dist.to_excel(excel_writer=excel, sheet_name=SheetName.Y_DIST.value)
        sr_slen_dist.to_excel(excel_writer=excel, sheet_name=SheetName.SLEN_DIST.value)
        df_metrics.to_excel(excel_writer=excel, sheet_name=SheetName.METRICS.value)
        df_word_weights_ranking.to_excel(excel_writer=excel, sheet_name=SheetName.WORD_WEIGHTS_RANKING.value)
        df_word_weights_result.to_excel(excel_writer=excel, sheet_name=SheetName.WORD_WEIGHTS_RESULT.value)
        df_sampled.to_excel(excel_writer=excel, sheet_name=SheetName.SAMPLING.value)
        df_mi.to_excel(excel_writer=excel, sheet_name=SheetName.MI.value)

    # model pipelineを保存
    with open(pipeline_dir.joinpath(FileName.PIPELINE.value), mode='wb') as fp:
        pickle.dump(pipeline_without_tokenizer, fp)

    logger.info('Output result Completed.')


def plot_label_dist(y: pd.Series) -> (go.Figure, pd.DataFrame):
    """
    目的変数のレコード数分布を計算・グラフ化する

    :param y: 目的変数
    :return: グラフ, ←の元データ
    """
    labels = sorted(y.unique())
    df_y_dist = y.value_counts().to_frame()
    df_y_dist['ratio'] = df_y_dist / df_y_dist.sum()  # レコード数の割合も算出
    # グラフの描写
    fig = px.bar(
        df_y_dist[y.name].reset_index().rename(columns={'index': 'Labels', y.name: 'Records'}),
        x='Labels', y='Records', color='Labels', category_orders={'Labels': labels},
        title=f'<i><b>{y.name} のラベルごとレコード数分布</b></i>'
    )
    fig.update_layout(margin=dict(t=50, l=100), yaxis=dict(tickformat=",.0"))

    return fig, df_y_dist


def plot_text_len_distribution(text_col: pd.Series) -> (go.Figure, pd.Series):
    """
    テキストカラムの文字数分布を計算・グラフ化する

    :param text_col: テキストカラム
    :return: グラフ, ←の元データ
    """
    sr_slen = text_col.apply(len)
    counts, bins = np.histogram(sr_slen, np.arange(0, sr_slen.max() + 10, 10))
    sr_slen_dist = pd.cut(sr_slen, bins=bins, right=False).value_counts().sort_index()

    fig = px.bar(
        x=bins[:-1], y=counts,
        labels={'x': f'{sr_slen.name}の1テキスト当たり文字数', 'y': 'Counts'},
        title=f'<i><b>{sr_slen.name} の文字数分布</b></i>'
    )
    fig.update_layout(margin=dict(t=50, l=100), yaxis=dict(tickformat=",.0"))

    return fig, sr_slen_dist


def plot_confusion_matrix(y_true: Iterable, y_pred: Iterable, normalize: bool = True, title: str = 'Confusion matrix',
                          cmap='Blues') -> go.FigureWidget:
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")

    classes = sorted(set(y_true))

    layout_heatmap = go.Layout(
        title=f'<i><b>{title}</b></i>',
        xaxis=dict(title='Predicted label'),
        yaxis=dict(title='True label', dtick=1)
    )

    ff_fig = ff.create_annotated_heatmap(
        x=classes, y=classes, z=cm,
        annotation_text=np.vectorize(lambda x: str(round(x, 3)))(cm),
        colorscale=cmap, showscale=True
    )

    fig = go.FigureWidget(ff_fig)
    fig.layout = layout_heatmap
    fig.layout.annotations = ff_fig.layout.annotations

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200))
    return fig


def classification_report_compared_with_random(y_true: Iterable, y_pred: Iterable, y_random: Iterable) -> pd.DataFrame:
    """
    評価指標について、モデリング結果とランダム分類器の結果とで比較した結果を出力する

    :param y_true: 真のラベル
    :param y_pred: 予測したラベル
    :param y_random: ランダムに予測したラベル
    :return: モデリング結果とランダム分類器の結果とで評価指標を比較したDataFrame
    """
    df_report_random = pd.DataFrame(classification_report(y_true=y_true, y_pred=y_random, output_dict=True)).T
    df_report = pd.DataFrame(classification_report(y_true=y_true, y_pred=y_pred, output_dict=True)).T

    report_dict = {'random': df_report_random, 'model': df_report}
    df_report_compared_with_random = (
        concat_df_with_MultiIndex_column(df_dict=report_dict, index=df_report.index)
            .swaplevel(0, 1, axis=1)[df_report.columns]
    )
    return df_report_compared_with_random


def concat_df_with_MultiIndex_column(df_dict: Dict[str, pd.DataFrame], index: pd.Index) -> pd.DataFrame:
    """
    複数のDataFrameをMultiIndexなcolumnでconcatする。以下のようなイメージ

    |      df_dict.keys()[0]      |      df_dict.keys()[1]      |
    | column1 | column2 | column3 | column1 | column2 | column3 |
    | ------- | ------- | ------- | ------- | ------- | ------- |
    | xxxx    | xxxx    | xxxx    | yyyy    | yyyy    | yyyy    |

    :param df_dict: {MultiIndexの第1層のカラム名: concatしたいDataFrame}
    :param index: concatしたDataFrameのindex
    :return: concatしたDataFrame
    """
    idx_labels = [[label] * len(df.columns) for label, df in df_dict.items()]
    idx_columns = [list(df.columns) for df in df_dict.values()]
    idx_report = pd.MultiIndex.from_arrays(
        list(map(lambda x: list(chain.from_iterable(x)), [idx_labels, idx_columns]))
    )
    df_concat = pd.DataFrame(pd.concat(df_dict.values(), axis=1).values, columns=idx_report, index=index)
    return df_concat


def plot_metric_compared_with_random(df_metrics: pd.DataFrame, metric: str = 'f1-score') -> go.FigureWidget:
    """
    指定の評価指標について、ランダムに予測した結果とモデリング結果を比較する
    全体の結果については`weighted avg`を採用している

    :param df_metrics: `classification_report_compared_with_random`の出力結果
    :param metric: 比較したい評価指標
    :return: 比較したグラフ
    """

    title = 'ランダム分類器との精度比較'
    df_plot = df_metrics.drop(index=['accuracy', 'macro avg']).rename(index={'weighted avg': '全体'})
    px_fig = px.bar(
        df_plot[metric].reset_index().melt(id_vars='index', value_vars=['random', 'model']),
        x='index', y='value', color='variable',
        text='value',
        barmode='group'
    )

    layout_bar = go.Layout(
        title=f'<i><b>{title}</b></i>',
        xaxis=dict(title='Labels'),
        yaxis=dict(dtick=0.1, tickformat=",.0%", showgrid=True)
    )
    fig = go.FigureWidget(px_fig)
    fig.layout = layout_bar

    fig.update_traces(texttemplate='%{text:.1%}', textposition='outside')
    fig.update_layout(margin=dict(t=50, l=100))

    return fig


def calc_words_wights(pipeline: Pipeline) -> (pd.DataFrame, pd.DataFrame):
    """
    単語の重みのランキングを返す
    ※議論の詳細はissue参照: https://gitlab.com/insight-tech-ds/common/classification-template/-/issues/2

    :param pipeline: 学習済みのpipeline
    :return: 単語の重みの計算結果と, そのランキング
    """
    model, vectorizer = pipeline.named_steps['model'], pipeline.named_steps['vectorizer']
    words, labels = vectorizer.get_feature_names(), model.classes_

    df_cnt = pd.DataFrame(model.feature_count_.T, index=words, columns=labels)  # 各単語×ラベルの出現数をカウント
    df_coef_ = pd.DataFrame(model.coef_.T, index=words, columns=labels)     # 各単語×ラベルの重み(=`feature_log_proba_`)を算出
    df_result = concat_df_with_MultiIndex_column(
        df_dict={
            'count': df_cnt, 'log_proba': df_coef_,
            'ratio_log_proba': df_coef_.div(df_coef_.sum(axis=1), axis=0)   # 重みのラベル間割合を算出
        },
        index=df_coef_.index
    )
    # heatmap用のランキングを作成
    df_ranking = create_ranking_df_with_labels(
        df_with_label_idx=df_result['ratio_log_proba'],
        label_col_name='word', value_col_name='ratio_log_proba', ascending=True
    )
    return df_result, df_ranking


def create_ranking_df_with_labels(df_with_label_idx: pd.DataFrame, label_col_name: str, value_col_name: str,
                                  ascending: bool) -> pd.DataFrame:
    """
    ランキングのheatmap用のDataFrameを作成する
    イメージはこんな感じ
    |        label_col_name       |       value_col_name        |
    | column1 | column2 | column3 | column1 | column2 | column3 |
    | ------- | ------- | ------- | ------- | ------- | ------- |
    | word1   | word2   | word1   | 99      | 77      | 44      |
    | word2   | word3   | word3   | 77      | 66      | 11      |

    :param df_with_label_idx: 表示したいラベルがindexに付いているDataFrame(valuesはヒートマップの色になる)
    :param label_col_name: 出力結果のランキングDataFrameのlabelの方のカラム名
    :param value_col_name: 出力結果のランキングDataFrameのvalue(heatmapの色)の方のカラム名
    :param ascending: ランキングを昇順にするか
    :return: 上記イメージのDataFrame。df_ranking['label_col_name']はheatmapのラベル、df_ranking['value_col_name']はheatmapの色になる
    """
    labels = list(df_with_label_idx.columns)

    # 値のDataFrameとラベルのDataFrameを別々に作成する(`sns.heatmap`に食わせるため)
    df_values, df_labels = pd.DataFrame(), pd.DataFrame()
    for label in labels:
        df_values[label] = df_with_label_idx[label].sort_values(ascending=ascending).values
        df_labels[label] = df_with_label_idx[label].sort_values(ascending=ascending).index
    # MultiIndexのcolumnにして返す(Excelにも出力するので、シートが分かれると2つのdfの関連が分かりづらくなるため)
    df_ranking = concat_df_with_MultiIndex_column(
        df_dict={label_col_name: df_labels, value_col_name: df_values},
        index=df_values.index + 1
    )
    return df_ranking


def plot_ranking(df_values: pd.DataFrame, df_labels: pd.DataFrame, rank_limit: int,
                 title: str, cmap: str = 'Reds_r') -> go.FigureWidget:
    """
    ランキングのheatmapを作成する

    :param df_values: heatmapの色になるDataFrame
    :param df_labels: heatmapのラベルになるDataFrame
    :param rank_limit: ランキングに上位何件を表示するか
    :param title: heatmapのタイトル
    :param cmap: カラースケール(matplotlibに準拠)
    :return: ランキングheatmapのグラフ
    """
    if set(df_values.index) != set(df_labels.index):
        raise Exception("Indexes between df_values and df_labels are different. Check both indexes.")
    df_values_to_show = df_values.head(rank_limit).sort_index(ascending=False)
    df_labels_to_show = df_labels.head(rank_limit).sort_index(ascending=False)

    labels = df_labels_to_show.columns
    rank_idx = list(df_labels_to_show.index)

    ff_fig = ff.create_annotated_heatmap(
        x=list(labels), y=rank_idx,
        z=df_values_to_show.astype(float).values,
        annotation_text=df_labels_to_show.values,
        colorscale=cmap
    )

    layout_heatmap = go.Layout(
        title=f'<i><b>{title}</b></i>',
        xaxis=dict(title='Labels'),
        yaxis=dict(title='Rank', dtick=1, type='category')
    )
    fig = go.FigureWidget(ff_fig)
    fig.layout = layout_heatmap
    fig.layout.annotations = ff_fig.layout.annotations

    fig.update_layout(
        margin=dict(t=50, l=200),
        width=130 * len(labels),
        height=1500
    )
    return fig


def sampling_predict_results(df_predicted: pd.DataFrame, label_col_name: str,
                             seed: int, n_sampling: int = 100) -> pd.DataFrame:
    """
    予測結果から、予測したラベル×真のラベルの組み合わせごとに指定の件数(満たない場合は全数)をサンプリングする

    :param df_predicted: 元csvファイルに予測結果がついたDataFrame
    :param label_col_name: 目的変数のカラム名
    :param seed: 乱数のseed
    :param n_sampling: ラベルの組み合わせごとにサンプリングする件数
    :return: サンプリングした結果
    """
    err_msg = 'Column `y_pred` not found. Before sampling results, fit the model and predict.'
    if 'y_pred' not in df_predicted.columns:
        raise Exception(err_msg)

    samples = []
    for true_label, pred_label in product(df_predicted[label_col_name].unique(), repeat=2):
        df_to_sample = \
            df_predicted[(df_predicted[label_col_name] == true_label) & (df_predicted['y_pred'] == pred_label)]
        len_to_sample = len(df_to_sample)

        df_sampled = \
            df_to_sample.sample(n=n_sampling, random_state=seed) if n_sampling < len_to_sample else df_to_sample
        samples.append(df_sampled)

    return pd.concat(samples, axis=0)


def calc_mi_between_label_and_word(vectorizer: CountVectorizer, X: Iterable, y: Iterable) -> pd.DataFrame:
    """
    カテゴリと単語のMIを計算する
    参考: https://aidiary.hatenablog.com/entry/20100619/1276950312

    :param vectorizer: CountVectorizer
    :param X: tokenize後のテキストカラム
    :param y: ラベルカラム
    :return: MI計算結果のDataFrame
    """
    logger.info('preprocessing...')
    feature_matrix = vectorizer.transform(X)

    le = LabelEncoder()
    y = le.fit_transform(y)
    csr_category = csr_matrix(y).T
    words = vectorizer.get_feature_names()
    word_ids = range(len(words))

    logger.info('calc MI elements...')
    n11 = reduce(
        lambda j, k: np.vstack([j, k]),
        [feature_matrix.multiply(csr_category == i).sum(axis=0).A1 for i in range(len(le.classes_))]
    ).T + 1  # スムージング
    n10 = reduce(
        lambda j, k: np.vstack([j, k]),
        [feature_matrix.multiply(csr_category != i).sum(axis=0).A1 for i in range(len(le.classes_))]
    ).T + 1  # スムージング
    with Pool(cpu_count()) as p:
        n01 = sum_category_per_target_word_excluded(
            pool=p, target_mat=n11,
            word_ids=word_ids
        ) + 1  # スムージング
        n00 = sum_category_per_target_word_excluded(
            pool=p, target_mat=n10,
            word_ids=word_ids
        ) + 1  # スムージング
    N = feature_matrix.sum()

    logger.info('calc MI...')
    temp1 = (n11 / N) * np.log2(N * n11 / ((n10 + n11) * (n01 + n11)))
    temp2 = (n01 / N) * np.log2(N * n01 / ((n00 + n01) * (n01 + n11)))
    temp3 = (n10 / N) * np.log2(N * n10 / ((n10 + n11) * (n00 + n10)))
    temp4 = (n00 / N) * np.log2(N * n00 / ((n00 + n01) * (n00 + n10)))
    score = temp1 + temp2 + temp3 + temp4

    df_score = pd.DataFrame(score, index=words, columns=le.classes_).fillna(0)
    logger.info('calc_mi_between_label_and_word Completed.')
    return df_score


def sum_category_per_target_word_excluded(pool: Pool, target_mat: np.ndarray, word_ids: Sequence) -> np.ndarray:
    arrays = pool.map(
        partial(filtering_matrix, target_mat=target_mat, word_ids=word_ids),
        tqdm(range(len(word_ids)))
    )
    return np.vstack([arrays])


def filtering_matrix(word_idx: int, target_mat: np.ndarray, word_ids: Sequence) -> np.ndarray:
    return target_mat[[x for x in word_ids if x != word_idx], :].sum(axis=0)
