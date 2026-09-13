import pytest

from ml.backtest import SplitConfig, split_timeline, walk_forward_folds


def test_days_mode_split_boundaries():
    split = split_timeline(40320, 1440, SplitConfig())
    assert split.train_end == 14 * 1440
    assert split.eval_start == 20160
    assert split.eval_end == 23 * 1440
    assert split.golden_start == 23 * 1440
    assert split.n == 40320


def test_days_mode_validates_coverage():
    with pytest.raises(ValueError, match="split days do not cover"):
        split_timeline(100, 1440, SplitConfig())


def test_fractions_mode_split():
    cfg = SplitConfig(fractions=(0.6, 0.3, 0.1), n_folds=10)
    split = split_timeline(1000, 1, cfg)
    assert (split.train_end, split.eval_start, split.eval_end, split.golden_start) == (
        600,
        600,
        900,
        900,
    )


def test_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1"):
        split_timeline(1000, 1, SplitConfig(fractions=(0.5, 0.3, 0.1), n_folds=5))


def test_walk_forward_folds_tile_eval_region_and_grow():
    split = split_timeline(40320, 1440, SplitConfig())
    embargo = 15
    folds = walk_forward_folds(split, embargo=embargo, cfg=SplitConfig(), points_per_day=1440)
    assert len(folds) == 9
    # eval windows tile the evaluation region exactly, in order
    assert folds[0].eval_start == split.eval_start
    assert folds[-1].eval_end == split.eval_end
    for prev, cur in zip(folds, folds[1:], strict=False):
        assert prev.eval_end == cur.eval_start
    # expanding window: training regions grow, never shrink
    train_ends = [f.train_end for f in folds]
    assert train_ends == sorted(train_ends)
    # every fold's training window ends exactly one embargo before its prediction window
    for fold in folds:
        assert fold.train_end == fold.eval_start - embargo
        assert fold.train_end <= fold.eval_start


def test_fold_training_targets_cannot_reach_evaluation_region():
    """With embargo == max horizon, train row t satisfies t + h < fold.eval_start."""
    split = split_timeline(40320, 1440, SplitConfig())
    max_horizon = 15
    folds = walk_forward_folds(split, embargo=max_horizon, cfg=SplitConfig(), points_per_day=1440)
    for fold in folds:
        last_train_row = fold.train_end - 1
        assert last_train_row + max_horizon < fold.eval_start


def test_embargo_zero_for_static_evaluation():
    split = split_timeline(40320, 1440, SplitConfig())
    folds = walk_forward_folds(split, embargo=0, cfg=SplitConfig(), points_per_day=1440)
    assert all(fold.embargo == 0 for fold in folds)
