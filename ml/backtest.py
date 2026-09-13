"""Timeline splitting and walk-forward fold generation.

No random splits exist anywhere in this project (PULSEGUARD.md §3.3). The timeline is
divided into three contiguous regions:

    [0, train_end)            initial training region (model fitting + threshold calibration)
    [eval_start, eval_end)    walk-forward evaluation region (Phase 2 metrics)
    [golden_start, n)         frozen golden holdout — never used for training, early
                              stopping, or threshold calibration; the only comparison
                              ground for champion vs challenger (Phase 4)

A walk-forward fold f fits on everything before its prediction window minus the embargo
gap, then predicts the window [t_c, t_c + step). The embargo equals the maximum forecast
horizon: training rows t must satisfy t + h < fold prediction start, so no training
target ever reaches into (or beyond) the evaluated region.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SplitConfig:
    # "days" mode: absolute region sizes in days of points_per_day rows (synthetic 1-min data).
    train_days: int = 14
    eval_days: int = 9
    golden_days: int = 5
    refit_every_days: int = 1
    # "fractions" mode: proportional splits for public datasets of any length/frequency.
    fractions: tuple[float, float, float] | None = None  # (train, eval, golden)
    n_folds: int = 10

    @property
    def mode(self) -> str:
        return "fractions" if self.fractions is not None else "days"


@dataclass(frozen=True)
class TimelineSplit:
    n: int
    train_end: int
    eval_start: int
    eval_end: int
    golden_start: int

    def validate(self) -> None:
        ordered = (
            0 < self.train_end <= self.eval_start < self.eval_end <= self.golden_start <= self.n
        )
        if not ordered:
            raise ValueError(f"invalid timeline split: {self}")


@dataclass(frozen=True)
class WalkForwardFold:
    fold_id: int
    train_start: int  # always 0: expanding window
    train_end: int  # exclusive; fitting rows [train_start, train_end)
    eval_start: int  # exclusive prediction-window start
    eval_end: int  # exclusive prediction-window end

    @property
    def embargo(self) -> int:
        return self.eval_start - self.train_end


def split_timeline(n: int, points_per_day: int, cfg: SplitConfig) -> TimelineSplit:
    if cfg.fractions is not None:
        train_frac, eval_frac, golden_frac = cfg.fractions
        if abs(train_frac + eval_frac + golden_frac - 1.0) > 1e-9:
            raise ValueError("split fractions must sum to 1")
        # accumulate per-region row counts to avoid float drift at the boundaries
        train_end = int(n * train_frac)
        eval_end = train_end + int(n * eval_frac)
        golden_start = eval_end
    else:
        train_end = cfg.train_days * points_per_day
        eval_end = train_end + cfg.eval_days * points_per_day
        golden_start = eval_end
        if train_end + cfg.eval_days * points_per_day + cfg.golden_days * points_per_day != n:
            raise ValueError(
                f"split days do not cover the series: n={n}, points_per_day={points_per_day}, "
                f"cfg={cfg.train_days}/{cfg.eval_days}/{cfg.golden_days}"
            )
    split = TimelineSplit(
        n=n, train_end=train_end, eval_start=train_end, eval_end=eval_end, golden_start=golden_start
    )
    split.validate()
    return split


def walk_forward_folds(
    split: TimelineSplit, embargo: int, cfg: SplitConfig, points_per_day: int
) -> list[WalkForwardFold]:
    if embargo < 0:
        raise ValueError("embargo must be non-negative")
    if cfg.fractions is not None:
        step = max(1, (split.eval_end - split.eval_start) // cfg.n_folds)
    else:
        step = max(1, cfg.refit_every_days * points_per_day)

    folds: list[WalkForwardFold] = []
    start = split.eval_start
    fold_id = 0
    while start < split.eval_end:
        end = min(start + step, split.eval_end)
        folds.append(
            WalkForwardFold(
                fold_id=fold_id,
                train_start=0,
                train_end=start - embargo,
                eval_start=start,
                eval_end=end,
            )
        )
        fold_id += 1
        start = end

    if not folds or folds[0].train_end <= folds[0].train_start:
        raise ValueError("walk-forward produced an empty first training window")
    return folds
