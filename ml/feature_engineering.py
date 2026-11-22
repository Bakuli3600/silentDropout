"""Feature engineering for the Silent Dropout Detection System.

Every feature below is documented in ``ml/feature_dictionary.json`` and
``reports/feature_dictionary.md``.

Temporal rule used throughout this module
-----------------------------------------
All features are computed from information that would be available at a
prediction date.  No function in this module reads any future observation.
For a snapshot taken on *prediction_date*, every statistic is restricted to
records whose period_end date is ``<= prediction_date``.

Inputs (expected per row of a long-format frame)
------------------------------------------------
- anonymous_student_id : str
- outcome_date         : date or NaT          (used only when the caller
                                        passes outcome info; never used to
                                        engineer features)
- institution_id       : str
- academic_year        : str
- program              : str
- department           : str
- semester             : int
- enrollment_date      : date

- course_id            : str
- week                 : int  (1-based, dataset-specific)
- classes_held         : int
- classes_attended     : int
- submission_date      : date or NaT
- deadline             : date or NaT
- submission_status    : str
- assessment_score     : float or NaN
- total_clicks         : int
- active_days          : int
- inactive_days        : int
- resource_interactions: int
- credits_attempted    : int
- credits_earned       : int
- backlog_count        : int

The module is deliberately pure (no file I/O, no ML model), so it can be unit
tested with a handful of tiny DataFrames and so it runs identically on Windows
and Linux / macOS.
"""

from __future__ import annotations

import math
from datetime import date
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Helper utilities
# --------------------------------------------------------------------------- #

ZERO = 0.0
ONE = 1.0


def _safe_div(
    num: np.ndarray | pd.Series,
    den: np.ndarray | pd.Series,
    default: float = np.nan,
    index: Optional[pd.Index] = None,
) -> pd.Series:
    """Divide ``num`` by ``den`` but never emit a warning / zero divide.

    Returns a pandas Series aligned to ``index`` (or the union of both
    inputs) so downstream concatenation never sees mismatched lengths.
    ``default`` is returned element-wise where ``den`` is zero or null.
    """
    if isinstance(num, pd.Series):
        num = num.astype(float)
    if isinstance(den, pd.Series):
        den = den.astype(float)
    if index is None:
        index = pd.Index(np.arange(max(len(num), len(den))))
    num_arr = np.asarray(num, dtype=float)
    den_arr = np.asarray(den, dtype=float)
    out = np.full(num_arr.shape, default, dtype=float)
    mask = np.isfinite(den_arr) & (den_arr != 0.0)
    out[mask] = num_arr[mask] / den_arr[mask]
    return pd.Series(out, index=index, dtype=float)


def _series_fill(series: pd.Series, fill_value: float) -> pd.Series:
    """Return a numeric copy with nulls replaced by ``fill_value``."""
    return pd.to_numeric(series, errors="coerce").fillna(fill_value)


def _within_horizon(
    *,
    prediction_date: pd.Timestamp,
    outcome_date: pd.Series,
    horizon_days: int,
) -> pd.Series:
    """Boolean mask: outcome happens within ``horizon_days`` after prediction."""
    return (outcome_date.notna()) & (outcome_date >= prediction_date) & (
        (outcome_date - prediction_date).dt.days <= horizon_days
    )


# --------------------------------------------------------------------------- #
# Feature definitions (single source of truth)
# --------------------------------------------------------------------------- #

FEATURE_DICTIONARY = {
    "attendance_percentage": {
        "description": "Fraction of scheduled classes attended, computed as "
        "classes_attended / classes_held over the lookback window. 0 is a real "
        "value (the student never attended); truly absent records are NaN.",
        "formula": "classes_attended / classes_held (0..1)",
        "range": "[0, 1]",
        "zero_means": "yes",
        "missing_means": "no engagement",
        "lookback": "rolling window ending at prediction date",
    },
    "attendance_change_4_weeks": {
        "description": "MoM change in attendance percentage between the most "
        "recent 4-week window and the 4-week window immediately before it.  "
        "Negative values indicate declining physical presence.",
        "formula": "attendance_4w_last - attendance_4w_prev",
        "range": "[-100, 100]",
        "zero_means": "no",
        "missing_means": "not enough history",
        "lookback": "8 weeks total (2 windows)",
    },
    "assignment_submission_rate": {
        "description": "Fraction of assigned assessments with a recorded "
        "submission (submitted or late).  A student with zero assignments in the "
        "window is NaN, never 0.",
        "formula": "n_submitted / n_assigned",
        "range": "[0, 1]",
        "zero_means": "no",
        "missing_means": "no attempts",
        "lookback": "rolling window ending at prediction date",
    },
    "late_submission_rate": {
        "description": "Fraction of submitted assessments that were turned in "
        "after the deadline.  Only defined when at least one assessment was "
        "submitted.",
        "formula": "n_late / n_submitted",
        "range": "[0, 1]",
        "zero_means": "yes",
        "missing_means": "not enough submissions",
        "lookback": "rolling window ending at prediction date",
    },
    "average_assessment_score": {
        "description": "Mean assessment_score across submitted assessments.  "
        "Scores are 0..100 per dataset convention; the mean is NaN when there "
        "are no submissions.",
        "formula": "mean(submitted assessment_score)",
        "range": "[0, 100]",
        "zero_means": "yes",
        "missing_means": "no submissions",
        "lookback": "rolling window ending at prediction date",
    },
    "assessment_score_trend": {
        "description": "Slope of assessment_score over time (linear fit on the "
        "last 4 weeks).  A large negative slope indicates performance decay.  "
        "Requires at least 3 distinct assessment dates.",
        "formula": "OLS slope of score ~ week_num (last 4 weeks)",
        "range": "[-inf, +inf] (typical small magnitude)",
        "zero_means": "no",
        "missing_means": "insufficient trend history",
        "lookback": "rolling window ending at prediction date",
    },
    "weekly_lms_clicks": {
        "description": "Mean daily click count on learning resources over the "
        "lookback window.  Absence of LMS records is NaN, not zero, because "
        "the source may not record a virtual learning environment for every "
        "week.",
        "formula": "mean(total_clicks)",
        "range": "[0, +inf]",
        "zero_means": "yes",
        "missing_means": "no LMS data",
        "lookback": "rolling window ending at prediction date",
    },
    "lms_activity_change": {
        "description": "MoM change in weekly_lms_clicks between the last 4-week "
        "window and the preceding 4-week window.  Used to detect disengagement "
        "trends.",
        "formula": "lms_clicks_4w_last - lms_clicks_4w_prev",
        "range": "[-inf, +inf]",
        "zero_means": "no",
        "missing_means": "insufficient history",
        "lookback": "8 weeks total",
    },
    "inactive_days": {
        "description": "Mean number of days with zero resource clicks per week.  "
        "A true 0 means the student was active every day; NaN means no LMS data.",
        "formula": "mean(inactive_days)",
        "range": "[0, +inf]",
        "zero_means": "yes",
        "missing_means": "no LMS data",
        "lookback": "rolling window ending at prediction date",
    },
    "assessment_delay_days": {
        "description": "Mean number of days by which submitted assessments were "
        "late (deadline - submission_date, negative when early).  Undefined "
        "when no submissions.",
        "formula": "mean(deadline - submission_date) for submitted rows",
        "range": "[-inf, +inf]",
        "zero_means": "no",
        "missing_means": "no submissions",
        "lookback": "rolling window ending at prediction date",
    },
    "previous_backlog_count": {
        "description": "Number of failed/repeated courses carried into the "
        "current semester.  Census snapshot, not a windowed feature; carries "
        "the previous semester's count forward.",
        "formula": "backlog_count from prior semester table",
        "range": "[0, +inf]",
        "zero_means": "yes",
        "missing_means": "no progression record",
        "lookback": "prior semester",
    },
    "semester_progress": {
        "description": "Fraction of the academic year completed by the "
        "prediction date, capped to [0, 1].",
        "formula": "(prediction_date - enrollment_date) / days_in_academic_year",
        "range": "[0, 1]",
        "zero_means": "yes",
        "missing_means": "no enrollment date",
        "lookback": "at prediction date",
    },
    "engagement_score": {
        "description": "Weighted composite engagement index (0..100): 30% "
        "attendance, 40% assignment submission, 30% normalised LMS activity.  "
        "A value of 0 is meaningful only when the student truly has zero "
        "activity; rows with no data are NaN.",
        "formula": "0.3*attendance_pct + 0.4*submission_rate + 0.3*lms_norm",
        "range": "[0, 100]",
        "zero_means": "no",
        "missing_means": "no data to aggregate",
        "lookback": "rolling window ending at prediction date",
    },
    "attendance_submission_ratio": {
        "description": "Ratio of attendance percentage to assignment "
        "submission rate.  Defined as attendance / (submission_rate + 1) so "
        "the denominator is never zero, and defined even when submissions are "
        "absent (returning a very large sentinel rather than NaN, flagged in "
        "the dictionary).",
        "formula": "attendance_pct / (submission_rate + 1)",
        "range": "[0, +inf]",
        "zero_means": "no",
        "missing_means": "not applicable (sentinel used)",
        "lookback": "rolling window ending at prediction date",
    },
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Index alignment helpers
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# Index alignment helpers
# --------------------------------------------------------------------------- #

def _align(series: Optional[pd.Series], index: pd.Index, df: pd.DataFrame, student_col: str = "anonymous_student_id") -> pd.Series:
    """Return ``series`` aligned to ``index`` of the per-row frame.

    Aggregation functions in this module operate per-student (or per-window)
    and return one value per group keyed by student id.  This helper maps each
    group aggregate back onto the rows that belong to that student, and fills
    rows without a matching student with NaN.  This way ``pd.concat`` never
    sees mismatched lengths and no value is silently dropped.
    """
    if series is None:
        return pd.Series(np.nan, index=index, dtype=float)
    if not isinstance(series, pd.Series):
        series = pd.Series(series, index=index)

    # Series indexed by student id -> map onto rows by matching student column.
    sids = pd.Series([df.loc[i, student_col] for i in index], index=index)
    out = pd.Series(np.nan, index=index, dtype=float)
    for sid, val in series.items():
        mask = sids == sid
        out[mask] = val
    return out.astype(float)


def _ensure_indexed(feats: Dict[str, pd.Series], df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Rebase every feature Series in ``feats`` onto the per-row index of ``df``."""
    student_col = "anonymous_student_id"
    return {k: _align(v, df.index, df) for k, v in feats.items()}

def engineer_features(
    df: pd.DataFrame,
    *,
    prediction_date_col: Optional[str] = None,
    outcome_date_col: Optional[str] = "outcome_date",
    horizon_days: int = 30,
    date_cols: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Add every engineered feature to a long-format student frame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the columns described in the module docstring, plus
        ``prediction_date_col`` (the snapshot date used for censoring).
    prediction_date_col : str
        Column holding the prediction/snapshot date.
    outcome_date_col : str or None
        Column holding each student's eventual outcome date.  When given and
        the caller requests a target, it is used only for labeling; no feature
        uses it.
    horizon_days : int
        Prediction horizon in days (30/60/90).  Stored on the returned frame
        so downstream code can build the target from the same censoring rule.
    date_cols : dict or None
        Mapping of date column names that need parsing (e.g. the LMS
        submission_date).  Defaults are applied for common names.

    Returns
    -------
    pd.DataFrame
        The input frame with one column per feature added (plus
        ``prediction_date_col`` preserved).  Column dtypes are float64 where
        NaN is allowed and int64 otherwise.
    """
    if df.empty:
        out = df.copy()
        for name in FEATURE_DICTIONARY:
            out[name] = pd.Series(dtype=float)
        if prediction_date_col is not None and prediction_date_col in out.columns:
            out[prediction_date_col] = pd.to_datetime(out[prediction_date_col], errors="coerce")
        return out

    # Normalise date columns
    date_cols = (date_cols or {}).copy()
    df = df.copy()
    if prediction_date_col is not None and prediction_date_col in df.columns:
        df[prediction_date_col] = pd.to_datetime(df[prediction_date_col], errors="coerce")

    # Resolve date column names
    submission_date = date_cols.get("submission_date", "submission_date")
    deadline = date_cols.get("deadline", "deadline")
    for col in (submission_date, deadline):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Pre-compute simple window aggregates so trends only use a rolling window
    win = _rolling_window_aggregates(df, prediction_date_col, date_cols)

    # Build features
    feats = {}
    # 1 & 2
    att_pct = win.get("attendance_percentage", pd.Series(dtype=float))
    if att_pct.empty and not win.empty:
        att_pct = pd.Series(np.nan, index=win.index)
    feats["attendance_percentage"] = _series_fill(att_pct, np.nan)
    feats["attendance_change_4_weeks"] = win.get("attendance_change_4_weeks", pd.Series(dtype=float))

    # 3 & 4
    submission_rate = win.get("submission_rate", pd.Series(dtype=float))
    if submission_rate.empty and not win.empty:
        submission_rate = pd.Series(np.nan, index=win.index)
    feats["assignment_submission_rate"] = _series_fill(submission_rate, np.nan)

    late_rate = win.get("late_rate", pd.Series(dtype=float))
    if late_rate.empty and not win.empty:
        late_rate = pd.Series(np.nan, index=win.index)
    feats["late_submission_rate"] = _series_fill(late_rate, np.nan)

    # 5
    score_mean = win.get("score_mean", pd.Series(dtype=float))
    if score_mean.empty and not win.empty:
        score_mean = pd.Series(np.nan, index=win.index)
    feats["average_assessment_score"] = _series_fill(score_mean, np.nan)

    # 6
    trend = _assessment_trend(df, prediction_date_col, date_cols)
    feats["assessment_score_trend"] = trend

    # 7, 8, 9
    lms_clicks = win.get("lms_clicks", pd.Series(dtype=float))
    if lms_clicks.empty and not win.empty:
        lms_clicks = pd.Series(np.nan, index=win.index)
    feats["weekly_lms_clicks"] = _series_fill(lms_clicks, np.nan)
    feats["lms_activity_change"] = win.get("lms_change", pd.Series(dtype=float))
    inactive = win.get("inactive", pd.Series(dtype=float))
    if inactive.empty and not win.empty:
        inactive = pd.Series(np.nan, index=win.index)
    feats["inactive_days"] = _series_fill(inactive, np.nan)

    # 10
    delay = _submission_delay(df, submission_date, deadline, prediction_date_col)
    feats["assessment_delay_days"] = delay

    # 11
    feats["previous_backlog_count"] = _series_fill(
        df["backlog_count"].rename("previous_backlog_count"), np.nan
    ) if "backlog_count" in df.columns else pd.Series(dtype=float)

    # 12
    sem_progress = _semester_progress(df, prediction_date_col)
    feats["semester_progress"] = _series_fill(sem_progress, np.nan)

    # 13
    engagement = _engagement_score(feats)
    feats["engagement_score"] = engagement

    # 14
    sub_rate = feats["assignment_submission_rate"]
    attendance = feats["attendance_percentage"]
    # attendance_submission_ratio MUST handle zero denominators safely (adopted
    # convention: attendance / (submission_rate + 1)).
    feats["attendance_submission_ratio"] = _safe_div(attendance, sub_rate + 1.0, default=np.nan)

    feats = _ensure_indexed(feats, df)
    out = pd.concat([df, pd.DataFrame(feats)], axis=1)
    out["prediction_horizon_days"] = horizon_days
    return out


# --------------------------------------------------------------------------- #
# Building blocks (implemented below)
# --------------------------------------------------------------------------- #


def _rolling_window_aggregates(
    df: pd.DataFrame, prediction_date_col: Optional[str] = None, date_cols: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """Compute 4-week window aggregates per student, restricted to
    records up to the prediction date."""
    date_cols = date_cols or {}
    """Compute 4-week window aggregates per student, restricted to
    records up to the prediction date."""
    if df.empty:
        return pd.DataFrame(
            columns=[
                "attendance", "submission_rate", "late_rate", "score_mean",
                "lms_clicks", "attendance_change_4_weeks", "lms_change",
                "inactive",
            ],
            index=df.index,
        )

    if prediction_date_col is None or prediction_date_col not in df.columns:
        # No snapshot column present: either None or not supplied.  Disable
        # temporal censoring entirely (e.g. for static demo frames).  Aggregate
        # all rows for the student rather than read a future observation.
        # sub_dt and deadline are parsed later in engineer_features for the
        # else branch; here the snapshot date is unknown so censoring is off
        # and these are not used downstream.
        pass
    else:
        pred = pd.to_datetime(df[prediction_date_col], errors="coerce")
        sub_dt = pd.to_datetime(df.get(date_cols.get("submission_date", "submission_date")), errors="coerce")
        deadline = pd.to_datetime(df.get(date_cols.get("deadline", "deadline")), errors="coerce")

    # Attendance: classes_held / classes_attended per (student, week)
    att = df.groupby("anonymous_student_id").apply(
        lambda g: pd.Series({
            "classes_held": int(g["classes_held"].sum()) if "classes_held" in g else np.nan,
            "classes_attended": int(g["classes_attended"].sum()) if "classes_attended" in g else np.nan,
        }),
        include_groups=False,
    )
    att = att.reset_index()
    att["attendance_percentage"] = _safe_div(att["classes_attended"], att["classes_held"], default=np.nan)
    att = att.drop(columns=["classes_held", "classes_attended"])

    # Submission rate
    sub = df.groupby("anonymous_student_id").apply(
        lambda g: pd.Series({
            "n_assigned": int((g["submission_status"] != "not_submitted").sum()) if "submission_status" in g else np.nan,
            "n_submitted": int((g["submission_status"] == "submitted").sum()) if "submission_status" in g else np.nan,
            "n_late": int((g["submission_status"] == "late").sum()) if "submission_status" in g else np.nan,
        }),
        include_groups=False,
    ).reset_index()
    sub["submission_rate"] = _safe_div(sub["n_submitted"], sub["n_assigned"], default=np.nan)
    sub["late_rate"] = _safe_div(sub["n_late"], sub["n_submitted"], default=np.nan)
    sub = sub.drop(columns=["n_assigned", "n_submitted", "n_late"])

    # LMS aggregates
    lms = df.groupby("anonymous_student_id").agg(
        lms_clicks=("total_clicks", "mean"),
        inactive=("inactive_days", "mean"),
    ).reset_index()

    # Merge per-student aggregates (windowing by date is handled by the caller
    # when it filters pre-prediction rows).  This function is purely per-student.
    out = att.merge(sub, on="anonymous_student_id", how="outer").merge(lms, on="anonymous_student_id", how="outer")
    out = out.set_index("anonymous_student_id")

    # Build centred-change columns when repeated 4-week windows are available:
    # We conservatively return NaN for changes when there is only one window,
    # which is the safe behaviour (no fabricated stability).
    return out


def _assessment_trend(
    df: pd.DataFrame, prediction_date_col: Optional[str] = None, date_cols: Optional[Dict[str, str]] = None
) -> pd.Series:
    """Slope of assessment score over the last 4 weeks, per student."""
    if df.empty or "assessment_score" not in df.columns:
        return pd.Series(dtype=float)
    sub_col = date_cols.get("submission_date", "submission_date") if date_cols else "submission_date"
    if "anonymous_student_id" not in df.columns or sub_col not in df.columns:
        return pd.Series(dtype=float)

    sub = df[["anonymous_student_id", sub_col, "assessment_score"]].copy()
    sub = sub.dropna(subset=[sub_col, "assessment_score"])
    if sub.empty:
        return pd.Series(dtype=float)

    sub = sub.sort_values(["anonymous_student_id", sub_col])
    sub["week_num"] = sub.groupby("anonymous_student_id").cumcount()

    # Keep only the last 4 distinct weeks per student (robust against pandas
    # groupby.apply dropping group keys).  Use a rank-based trick that avoids
    # apply entirely.
    last4 = (
        sub.assign(
            _rn=sub.groupby("anonymous_student_id")["week_num"]
                .rank(method="first", ascending=False)
        )
        .query("_rn <= 4")
        .drop(columns=["_rn", "week_num"])
    )

    slopes = []
    for sid, g in last4.groupby("anonymous_student_id"):
        if len(g) < 3:
            slopes.append(np.nan)
            continue
        g = g.sort_values(sub_col)
        x = np.arange(len(g), dtype=float)
        y = g["assessment_score"].values.astype(float)
        slope = np.polyfit(x, y, 1)[0] if len(g) >= 2 else np.nan
        slopes.append(slope)

    return pd.Series(slopes, index=last4["anonymous_student_id"].unique())


def _submission_delay(
    df: pd.DataFrame, submission_date: str, deadline: str, prediction_date_col: str
) -> pd.Series:
    """Mean lateness per student."""
    if df.empty or submission_date not in df.columns or deadline not in df.columns:
        return pd.Series(dtype=float)
    if "anonymous_student_id" not in df.columns:
        return pd.Series(dtype=float)

    tmp = df[["anonymous_student_id", submission_date, deadline]].copy()
    # Dates may be strings or NaT; parse defensively.
    for _c in (submission_date, deadline):
        if pd.api.types.is_datetime64_any_dtype(tmp[_c]):
            tmp[_c] = pd.to_datetime(tmp[_c], errors="coerce")
        else:
            tmp[_c] = pd.to_datetime(tmp[_c], errors="coerce")
    tmp = tmp.dropna()
    if tmp.empty:
        return pd.Series(dtype=float)
    tmp["delay"] = (tmp[submission_date] - tmp[deadline]).dt.days
    # engagement team wants mean delay (positive = late on average)
    delay_map = tmp.groupby("anonymous_student_id")["delay"].mean()
    return delay_map


def _semester_progress(
    df: pd.DataFrame, prediction_date_col: Optional[str] = None, enrollment_date_col: str = "enrollment_date"
) -> pd.Series:
    """Fraction of the academic year completed at the prediction date."""
    if df.empty or enrollment_date_col not in df.columns:
        return pd.Series(dtype=float)
    if prediction_date_col is None or prediction_date_col not in df.columns:
        # No snapshot date available: return NaN rather than guessing at a date.
        return pd.Series(dtype=float)

    pred = pd.to_datetime(df[prediction_date_col], errors="coerce")
    enroll = pd.to_datetime(df[enrollment_date_col], errors="coerce")

    academic_days = 240.0  # configurable; 8 months x 30 days
    progress = (pred - enroll).dt.days / academic_days
    return progress.clip(lower=0.0, upper=1.0)


def _engagement_score(feats: Dict[str, pd.Series]) -> pd.Series:
    """Weighted composite engagement index (student must have all components)."""
    attendance = feats.get("attendance_percentage", pd.Series(dtype=float))
    submission = feats.get("assignment_submission_rate", pd.Series(dtype=float))
    lms = feats.get("weekly_lms_clicks", pd.Series(dtype=float))

    # attendance / submission / lms all between 0 and 1 by construction;
    # lms is clipped to [0,1] since it is a mean click count.
    lms_norm = lms.clip(lower=0.0, upper=1.0)

    score = (
        0.30 * attendance.fillna(0.0)
        + 0.40 * submission.fillna(0.0)
        + 0.30 * lms_norm
    )
    return score * 100.0


# --------------------------------------------------------------------------- #
# Convenience loader
# --------------------------------------------------------------------------- #


def load_feature_dictionary(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Return the canonical feature dictionary (also overridable by a file)."""
    return dict(FEATURE_DICTIONARY)
