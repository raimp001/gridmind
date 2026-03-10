"""Statistical significance testing for metric improvements.

Answers the question: "Is this improvement real or just noise?"

Implements:
- Welch's t-test for comparing means across experiment groups
- Confidence intervals for metric estimates
- Effect size (Cohen's d) for practical significance
- Minimum sample size estimation
- No external dependencies (pure Python + math stdlib)
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class SignificanceResult:
    """Result of a significance test between two groups."""

    is_significant: bool
    p_value: float
    confidence_level: float  # e.g. 0.95
    effect_size: float  # Cohen's d
    effect_label: str  # "negligible", "small", "medium", "large"
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    n_a: int
    n_b: int
    improvement: float  # relative improvement (b - a) / |a|
    confidence_interval: tuple[float, float]  # 95% CI for difference

    def summary(self) -> str:
        direction = "improved" if self.improvement > 0 else "declined"
        return (
            f"{direction} by {abs(self.improvement):.1%} "
            f"(p={self.p_value:.4f}, d={self.effect_size:.2f} [{self.effect_label}], "
            f"n={self.n_a}+{self.n_b}, "
            f"{'SIGNIFICANT' if self.is_significant else 'not significant'})"
        )


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _variance(values: list[float], ddof: int = 1) -> float:
    """Sample variance with Bessel's correction (ddof=1)."""
    if len(values) <= ddof:
        return 0.0
    m = _mean(values)
    return sum((x - m) ** 2 for x in values) / (len(values) - ddof)


def _std(values: list[float], ddof: int = 1) -> float:
    return math.sqrt(_variance(values, ddof))


def _t_cdf_approx(t: float, df: float) -> float:
    """Approximate the CDF of the t-distribution.

    Uses the regularized incomplete beta function approximation.
    Accurate enough for our purposes (significance testing, not publishing papers).
    """
    if df <= 0:
        return 0.5

    # For large df, approximate with normal distribution
    if df > 100:
        return _normal_cdf(t)

    # Use the relationship: CDF(t, df) = 1 - 0.5 * I(df/(df+t^2), df/2, 1/2)
    # where I is the regularized incomplete beta function
    x = df / (df + t * t)
    a = df / 2.0
    b = 0.5

    # Compute regularized incomplete beta using continued fraction
    beta_val = _incomplete_beta(x, a, b)

    if t >= 0:
        return 1.0 - 0.5 * beta_val
    else:
        return 0.5 * beta_val


def _normal_cdf(x: float) -> float:
    """Approximate CDF of standard normal using error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularized incomplete beta function I_x(a, b).

    Uses the continued fraction representation (Lentz's algorithm).
    """
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0

    # Use symmetry relation for better convergence
    if x > (a + 1) / (a + b + 2):
        return 1.0 - _incomplete_beta(1 - x, b, a)

    # Log of the prefactor
    try:
        ln_prefactor = (
            math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
            + a * math.log(x) + b * math.log(1 - x)
        )
        prefactor = math.exp(ln_prefactor)
    except (ValueError, OverflowError):
        return 0.5

    # Continued fraction (Lentz's method)
    eps = 1e-30
    max_iter = 200

    # Modified Lentz's algorithm
    f = 1.0 + eps
    c = f
    d = 0.0

    for m in range(0, max_iter):
        if m == 0:
            alpha_m = 1.0
        elif m % 2 == 1:
            k = (m - 1) // 2 + 1
            alpha_m = -(a + k - 1) * (a + b + k - 1) * x / ((a + 2 * k - 2) * (a + 2 * k - 1))
        else:
            k = m // 2
            alpha_m = k * (b - k) * x / ((a + 2 * k - 1) * (a + 2 * k))

        d = 1.0 + alpha_m * d
        if abs(d) < eps:
            d = eps
        d = 1.0 / d

        c = 1.0 + alpha_m / c
        if abs(c) < eps:
            c = eps

        delta = c * d
        f *= delta

        if abs(delta - 1.0) < 1e-10:
            break

    return prefactor * (f - 1.0) / a


def welch_t_test(
    group_a: list[float],
    group_b: list[float],
    confidence: float = 0.95,
) -> SignificanceResult:
    """Welch's t-test for unequal variances.

    Tests whether the means of two groups are significantly different.

    Args:
        group_a: Metric values from the baseline/control group.
        group_b: Metric values from the treatment group.
        confidence: Confidence level (default 0.95 = 95%).

    Returns:
        SignificanceResult with test statistics and interpretation.
    """
    n_a = len(group_a)
    n_b = len(group_b)

    if n_a < 2 or n_b < 2:
        return SignificanceResult(
            is_significant=False,
            p_value=1.0,
            confidence_level=confidence,
            effect_size=0.0,
            effect_label="insufficient_data",
            mean_a=_mean(group_a) if group_a else 0.0,
            mean_b=_mean(group_b) if group_b else 0.0,
            std_a=0.0,
            std_b=0.0,
            n_a=n_a,
            n_b=n_b,
            improvement=0.0,
            confidence_interval=(0.0, 0.0),
        )

    mean_a = _mean(group_a)
    mean_b = _mean(group_b)
    var_a = _variance(group_a)
    var_b = _variance(group_b)
    std_a = math.sqrt(var_a)
    std_b = math.sqrt(var_b)

    # Standard error of the difference
    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se < 1e-15:
        # No variance = can't test
        return SignificanceResult(
            is_significant=False,
            p_value=1.0,
            confidence_level=confidence,
            effect_size=0.0,
            effect_label="no_variance",
            mean_a=mean_a,
            mean_b=mean_b,
            std_a=std_a,
            std_b=std_b,
            n_a=n_a,
            n_b=n_b,
            improvement=0.0,
            confidence_interval=(0.0, 0.0),
        )

    # t-statistic
    t_stat = (mean_b - mean_a) / se

    # Welch-Satterthwaite degrees of freedom
    numerator = (var_a / n_a + var_b / n_b) ** 2
    denominator = (
        (var_a / n_a) ** 2 / (n_a - 1)
        + (var_b / n_b) ** 2 / (n_b - 1)
    )
    if denominator < 1e-15:
        df = n_a + n_b - 2
    else:
        df = numerator / denominator

    # Two-tailed p-value
    p_value = 2.0 * (1.0 - _t_cdf_approx(abs(t_stat), df))
    p_value = max(0.0, min(1.0, p_value))

    # Effect size (Cohen's d)
    pooled_std = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std > 1e-15:
        cohens_d = (mean_b - mean_a) / pooled_std
    else:
        cohens_d = 0.0

    # Effect size label
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        effect_label = "negligible"
    elif abs_d < 0.5:
        effect_label = "small"
    elif abs_d < 0.8:
        effect_label = "medium"
    else:
        effect_label = "large"

    # Confidence interval for the difference
    alpha = 1.0 - confidence
    # Approximate critical value from t-distribution
    # For df > 30, use normal approximation
    if df > 30:
        z_crit = _normal_ppf(1.0 - alpha / 2.0)
    else:
        z_crit = _t_ppf_approx(1.0 - alpha / 2.0, df)

    diff = mean_b - mean_a
    margin = z_crit * se
    ci = (diff - margin, diff + margin)

    # Relative improvement
    if abs(mean_a) > 1e-15:
        improvement = (mean_b - mean_a) / abs(mean_a)
    else:
        improvement = 0.0

    return SignificanceResult(
        is_significant=p_value < (1 - confidence),
        p_value=p_value,
        confidence_level=confidence,
        effect_size=cohens_d,
        effect_label=effect_label,
        mean_a=mean_a,
        mean_b=mean_b,
        std_a=std_a,
        std_b=std_b,
        n_a=n_a,
        n_b=n_b,
        improvement=improvement,
        confidence_interval=ci,
    )


def _normal_ppf(p: float) -> float:
    """Approximate inverse normal CDF (percent point function).

    Uses the rational approximation from Abramowitz & Stegun.
    """
    if p <= 0:
        return -10.0
    if p >= 1:
        return 10.0
    if p == 0.5:
        return 0.0

    if p < 0.5:
        return -_normal_ppf(1 - p)

    # Rational approximation for 0.5 < p < 1
    t = math.sqrt(-2.0 * math.log(1.0 - p))

    # Coefficients
    c0 = 2.515517
    c1 = 0.802853
    c2 = 0.010328
    d1 = 1.432788
    d2 = 0.189269
    d3 = 0.001308

    return t - (c0 + c1 * t + c2 * t * t) / (1 + d1 * t + d2 * t * t + d3 * t * t * t)


def _t_ppf_approx(p: float, df: float) -> float:
    """Approximate inverse t-distribution CDF.

    Uses normal approximation with correction for small df.
    """
    z = _normal_ppf(p)
    # Cornish-Fisher expansion for better accuracy at low df
    g1 = (z ** 3 + z) / (4 * df)
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)
    return z + g1 + g2


def minimum_sample_size(
    effect_size: float = 0.5,
    power: float = 0.8,
    alpha: float = 0.05,
) -> int:
    """Estimate minimum sample size per group for a two-sample t-test.

    Args:
        effect_size: Expected Cohen's d (0.2=small, 0.5=medium, 0.8=large).
        power: Desired statistical power (default 0.8 = 80%).
        alpha: Significance level (default 0.05 = 5%).

    Returns:
        Minimum number of observations needed per group.
    """
    if effect_size <= 0:
        return 1000  # Can't detect zero effect

    z_alpha = _normal_ppf(1 - alpha / 2)
    z_beta = _normal_ppf(power)

    n = ((z_alpha + z_beta) / effect_size) ** 2
    return max(2, math.ceil(n))


def confidence_interval(
    values: list[float],
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """Compute confidence interval for a sample mean.

    Returns:
        (lower, mean, upper) bounds.
    """
    n = len(values)
    if n < 2:
        m = values[0] if values else 0.0
        return (m, m, m)

    m = _mean(values)
    se = _std(values) / math.sqrt(n)
    z = _normal_ppf(1 - (1 - confidence) / 2)
    margin = z * se
    return (m - margin, m, m + margin)


def metric_is_improving(
    history: list[float],
    window: int = 10,
    confidence: float = 0.90,
) -> SignificanceResult | None:
    """Test whether the metric is improving over time.

    Splits the history into early and recent windows and tests
    whether the recent window is significantly better.

    Returns None if insufficient data.
    """
    if len(history) < 2 * window:
        return None

    early = history[:window]
    recent = history[-window:]
    return welch_t_test(early, recent, confidence)
