from __future__ import annotations

import math
from collections import Counter
from typing import Iterable

import pandas as pd

from freewill.action_canonicalization import add_action_canonical_columns, normalize_action_text


def shannon_entropy(values: Iterable[object]) -> float:
    clean = [str(value) for value in values if str(value) != ""]
    if not clean:
        return 0.0
    total = len(clean)
    counts = Counter(clean)
    return float(-sum((count / total) * math.log2(count / total) for count in counts.values()))


def semantic_cluster_labels(values: Iterable[object], *, threshold: float = 0.90) -> list[str]:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [normalize_action_text(value) for value in values]
    if not texts:
        return []
    non_empty = [text if text else "missing_action" for text in texts]
    if len(set(non_empty)) == 1:
        return ["cluster_000"] * len(non_empty)
    try:
        vectors = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1).fit_transform(non_empty)
        similarities = cosine_similarity(vectors)
    except ValueError:
        return [f"cluster_{index:03d}" for index, _ in enumerate(non_empty)]

    labels = [-1] * len(non_empty)
    cluster_id = 0
    for index in range(len(non_empty)):
        if labels[index] != -1:
            continue
        labels[index] = cluster_id
        for other in range(index + 1, len(non_empty)):
            if labels[other] == -1 and float(similarities[index, other]) >= threshold:
                labels[other] = cluster_id
        cluster_id += 1
    return [f"cluster_{label:03d}" for label in labels]


def entropy_summary(
    frame: pd.DataFrame,
    *,
    group_columns: list[str],
    action_column: str = "final_action",
    semantic_threshold: float = 0.90,
) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = add_action_canonical_columns(frame, action_column=action_column)
    rows: list[dict[str, object]] = []
    for keys, group in working.groupby(group_columns, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        semantic_labels = semantic_cluster_labels(group[action_column].fillna("").astype(str), threshold=semantic_threshold)
        row = {column: value for column, value in zip(group_columns, keys)}
        row.update(
            {
                "row_count": int(len(group)),
                "unique_raw_action_count": int(group["raw_action"].nunique()),
                "unique_canonical_action_count": int(group["canonical_action_rule"].nunique()),
                "unique_semantic_cluster_count": int(len(set(semantic_labels))),
                "unique_action_family_count": int(group["action_family"].nunique()),
                "H_raw_string": shannon_entropy(group["raw_action"]),
                "H_canonical_rule": shannon_entropy(group["canonical_action_rule"]),
                "H_semantic_cluster": shannon_entropy(semantic_labels),
                "H_action_family": shannon_entropy(group["action_family"]),
                "semantic_cluster_method": f"tfidf_char_cosine_threshold_{semantic_threshold:.2f}",
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)

