import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pytest
from attr import asdict
from attrs import define, field

from tools.auto_labeling_3d.filter_objects.ensemble.ensemble_model import (
    EnsembleModel,
)


@define
class PredInstance3D:
    """3D予測インスタンスのデータクラス。"""

    bbox_3d: List[float]
    velocity: List[float]
    instance_id_3d: str
    bbox_label_3d: int
    bbox_score_3d: float

    def asdict(self) -> Dict:
        """オブジェクトを辞書に変換します。"""
        return asdict(self)


# テスト用のフィクスチャ
@pytest.fixture
def ensemble_model():
    """テスト用のEnsembleModelインスタンスを返します。"""
    ensemble_setting = {
        "weights": [0.7, 0.3],
        "iou_threshold": 0.5,
        "skip_box_threshold": 0.2,
    }
    logger = logging.getLogger("test_ensemble")
    return EnsembleModel(
        ensemble_setting=ensemble_setting,
        logger=logger,
    )


@pytest.fixture
def pred_instances():
    """テスト用の予測インスタンスのリストを返します。"""
    return [
        PredInstance3D(
            bbox_3d=[1.0, 2.0, 3.0, 2.0, 2.0, 2.0, 0.0],
            velocity=[1.0, 0.0],
            instance_id_3d="1",
            bbox_label_3d=0,
            bbox_score_3d=0.9,
        ),
        PredInstance3D(
            bbox_3d=[1.1, 2.1, 3.1, 2.0, 2.0, 2.0, 0.0],
            velocity=[0.0, 1.0],
            instance_id_3d="2",
            bbox_label_3d=0,
            bbox_score_3d=0.8,
        ),
        PredInstance3D(
            bbox_3d=[10.0, 10.0, 3.0, 2.0, 2.0, 2.0, 0.0],
            velocity=[1.0, 0.0],
            instance_id_3d="3",
            bbox_label_3d=1,
            bbox_score_3d=0.85,
        ),
        PredInstance3D(
            bbox_3d=[10.1, 10.1, 3.0, 2.0, 2.0, 2.0, 0.0],
            velocity=[0.0, 1.0],
            instance_id_3d="4",
            bbox_label_3d=1,
            bbox_score_3d=0.75,
        ),
    ]


@pytest.fixture
def create_result_with_instances():
    """指定したインスタンスでテスト結果を作成する関数を返します。"""

    def _create_result(instances: List[PredInstance3D], metainfo: Optional[Dict] = None) -> Dict:
        """指定したインスタンスでテスト結果を作成します。"""
        return {
            "metainfo": metainfo or {"test": "data"},
            "data_list": [
                {
                    "frame_id": 0,
                    "pred_instances_3d": [instance.asdict() for instance in instances],
                }
            ],
        }

    return _create_result


def test_ensemble_single_result(ensemble_model, pred_instances, create_result_with_instances):
    """単一結果のアンサンブルをテストします。"""
    # 単一の結果を作成
    result = create_result_with_instances([pred_instances[0]])
    # アンサンブルは入力結果を変更せずに返すはず
    assert ensemble_model.ensemble([result]) == result


def test_ensemble_multiple_results(ensemble_model, pred_instances, create_result_with_instances):
    """複数結果のアンサンブルをテストします。"""
    # テストデータを作成
    results = [
        create_result_with_instances([pred_instances[0]]),
        create_result_with_instances([pred_instances[1]], metainfo={"test": "data2"}),
    ]

    # アンサンブルメソッドを呼び出す
    result = ensemble_model.ensemble(results)
    pred_instances_result = result["data_list"][0]["pred_instances_3d"]

    # metainfoは変更しない
    assert result["metainfo"] == results[0]["metainfo"]
    assert len(result["data_list"]) == 1
    assert "pred_instances_3d" in result["data_list"][0]

    # NMS後は1つのボックスのみ残るはず
    assert len(pred_instances_result) == 1

    # 最も高い重み付きスコアのボックスが保持されるはず
    # モデル1: 0.9 * 0.7 = 0.63
    # モデル2: 0.8 * 0.3 = 0.24
    # モデル1のほうが高い重み付きスコアを持つので、そのボックスが保持されるはず
    assert pytest.approx(pred_instances_result[0]["bbox_score_3d"], abs=1e-5) == 0.63
