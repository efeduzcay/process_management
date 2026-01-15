"""
Statistical Analysis Module
---------------------------
Rigorous statistical testing for MG-RR vs RR comparison.

Includes:
- Descriptive statistics (mean, median, variance, std)
- Paired T-Test (for matched samples)
- Cohen's d effect size
- Confidence intervals

Author: MG-RR Research Study
"""

import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TTestResult:
    """Results from a paired t-test."""
    t_statistic: float
    p_value: float
    degrees_of_freedom: int
    mean_difference: float
    std_error: float
    confidence_interval_95: Tuple[float, float]
    
    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if result is statistically significant at given alpha level."""
        return self.p_value < alpha
    
    def __str__(self) -> str:
        sig = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else ""
        return (
            f"t({self.degrees_of_freedom}) = {self.t_statistic:.4f}, "
            f"p = {self.p_value:.6f}{sig}\n"
            f"Mean diff = {self.mean_difference:.4f}, SE = {self.std_error:.4f}\n"
            f"95% CI: [{self.confidence_interval_95[0]:.4f}, {self.confidence_interval_95[1]:.4f}]"
        )


@dataclass
class EffectSize:
    """Effect size measurements."""
    cohens_d: float
    interpretation: str  # "negligible", "small", "medium", "large"
    
    def __str__(self) -> str:
        return f"Cohen's d = {self.cohens_d:.4f} ({self.interpretation})"


@dataclass
class DescriptiveStats:
    """Descriptive statistics for a dataset."""
    n: int
    mean: float
    median: float
    variance: float
    std_dev: float
    min_val: float
    max_val: float
    q1: float  # 25th percentile
    q3: float  # 75th percentile
    iqr: float  # Interquartile range
    
    def __str__(self) -> str:
        return (
            f"n={self.n}, mean={self.mean:.4f}, median={self.median:.4f}\n"
            f"std={self.std_dev:.4f}, var={self.variance:.4f}\n"
            f"range=[{self.min_val:.4f}, {self.max_val:.4f}]\n"
            f"IQR=[{self.q1:.4f}, {self.q3:.4f}], iqr={self.iqr:.4f}"
        )


class StatisticalAnalysis:
    """
    Statistical analysis tools for Monte Carlo simulation results.
    
    Uses pure Python implementations (no external dependencies like scipy)
    for portability, with formulas from standard statistics textbooks.
    """
    
    @staticmethod
    def mean(data: List[float]) -> float:
        """Calculate arithmetic mean."""
        if not data:
            return 0.0
        return sum(data) / len(data)
    
    @staticmethod
    def variance(data: List[float], ddof: int = 1) -> float:
        """
        Calculate sample variance.
        
        Args:
            data: List of values
            ddof: Delta degrees of freedom (1 for sample variance, 0 for population)
        """
        if len(data) <= ddof:
            return 0.0
        m = StatisticalAnalysis.mean(data)
        return sum((x - m) ** 2 for x in data) / (len(data) - ddof)
    
    @staticmethod
    def std_dev(data: List[float], ddof: int = 1) -> float:
        """Calculate standard deviation."""
        return math.sqrt(StatisticalAnalysis.variance(data, ddof))
    
    @staticmethod
    def median(data: List[float]) -> float:
        """Calculate median."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        n = len(sorted_data)
        mid = n // 2
        if n % 2 == 0:
            return (sorted_data[mid - 1] + sorted_data[mid]) / 2
        return sorted_data[mid]
    
    @staticmethod
    def percentile(data: List[float], p: float) -> float:
        """
        Calculate percentile using linear interpolation.
        
        Args:
            data: List of values
            p: Percentile (0-100)
        """
        if not data:
            return 0.0
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        # Calculate rank
        rank = (p / 100) * (n - 1)
        lower_idx = int(rank)
        upper_idx = min(lower_idx + 1, n - 1)
        fraction = rank - lower_idx
        
        return sorted_data[lower_idx] + fraction * (sorted_data[upper_idx] - sorted_data[lower_idx])
    
    @staticmethod
    def descriptive_stats(data: List[float]) -> DescriptiveStats:
        """
        Calculate comprehensive descriptive statistics.
        
        Args:
            data: List of values
            
        Returns:
            DescriptiveStats object with all metrics
        """
        if not data:
            return DescriptiveStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        n = len(data)
        m = StatisticalAnalysis.mean(data)
        med = StatisticalAnalysis.median(data)
        var = StatisticalAnalysis.variance(data)
        std = math.sqrt(var)
        
        q1 = StatisticalAnalysis.percentile(data, 25)
        q3 = StatisticalAnalysis.percentile(data, 75)
        
        return DescriptiveStats(
            n=n,
            mean=m,
            median=med,
            variance=var,
            std_dev=std,
            min_val=min(data),
            max_val=max(data),
            q1=q1,
            q3=q3,
            iqr=q3 - q1
        )
    
    @staticmethod
    def _t_cdf(t: float, df: int) -> float:
        """
        Approximate the CDF of Student's t-distribution.
        
        Uses the regularized incomplete beta function approximation.
        This is a simplified implementation for p-value calculation.
        
        Args:
            t: t-statistic
            df: degrees of freedom
            
        Returns:
            Approximate CDF value
        """
        # For large df, approximate with normal distribution
        if df > 100:
            # Standard normal CDF approximation (Abramowitz and Stegun)
            x = t
            a1 = 0.254829592
            a2 = -0.284496736
            a3 = 1.421413741
            a4 = -1.453152027
            a5 = 1.061405429
            p = 0.3275911
            
            sign = 1 if x >= 0 else -1
            x = abs(x) / math.sqrt(2)
            
            t_val = 1.0 / (1.0 + p * x)
            y = 1.0 - (((((a5 * t_val + a4) * t_val) + a3) * t_val + a2) * t_val + a1) * t_val * math.exp(-x * x)
            
            return 0.5 * (1.0 + sign * y)
        
        # For smaller df, use regularized incomplete beta function
        # Approximation using continued fraction
        x = df / (df + t * t)
        
        # Regularized incomplete beta function I_x(a, b) where a = df/2, b = 0.5
        a = df / 2
        b = 0.5
        
        # Simple approximation for beta function
        if t >= 0:
            return 1 - 0.5 * StatisticalAnalysis._beta_inc(x, a, b)
        else:
            return 0.5 * StatisticalAnalysis._beta_inc(x, a, b)
    
    @staticmethod
    def _beta_inc(x: float, a: float, b: float, max_iter: int = 200) -> float:
        """
        Regularized incomplete beta function approximation.
        
        Uses Lentz's continued fraction algorithm.
        """
        if x == 0:
            return 0
        if x == 1:
            return 1
        
        # Use continued fraction representation
        tiny = 1e-30
        
        # Compute ln(B(a,b)) using log gamma approximation
        def log_gamma(z):
            """Stirling's approximation for log gamma."""
            if z < 0.5:
                return math.log(math.pi / math.sin(math.pi * z)) - log_gamma(1 - z)
            z -= 1
            x = 0.99999999999980993
            for i, c in enumerate([
                676.5203681218851, -1259.1392167224028, 771.32342877765313,
                -176.61502916214059, 12.507343278686905, -0.13857109526572012,
                9.9843695780195716e-6, 1.5056327351493116e-7
            ]):
                x += c / (z + i + 1)
            t = z + 7.5
            return 0.5 * math.log(2 * math.pi) + (z + 0.5) * math.log(t) - t + math.log(x)
        
        log_beta = log_gamma(a) + log_gamma(b) - log_gamma(a + b)
        front = math.exp(a * math.log(x) + b * math.log(1 - x) - log_beta) / a
        
        # Lentz's algorithm for continued fraction
        f = 1.0
        c = 1.0
        d = 0.0
        
        for m in range(1, max_iter + 1):
            m2 = 2 * m
            
            # Even step
            aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
            d = 1 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1 / d
            f *= c * d
            
            # Odd step
            aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
            d = 1 + aa * d
            if abs(d) < tiny:
                d = tiny
            c = 1 + aa / c
            if abs(c) < tiny:
                c = tiny
            d = 1 / d
            delta = c * d
            f *= delta
            
            if abs(delta - 1) < 1e-10:
                break
        
        return front * f
    
    def paired_ttest(
        self, 
        sample1: List[float], 
        sample2: List[float],
        alternative: str = 'two-sided'
    ) -> TTestResult:
        """
        Perform paired t-test.
        
        Tests whether the mean difference between paired samples is zero.
        
        H0: μ_diff = 0 (no difference between algorithms)
        H1: μ_diff ≠ 0 (two-sided) or μ_diff > 0 or μ_diff < 0 (one-sided)
        
        Args:
            sample1: First sample (e.g., RR results)
            sample2: Second sample (e.g., MG-RR results)  
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            TTestResult with t-statistic, p-value, and confidence interval
        """
        if len(sample1) != len(sample2):
            raise ValueError("Samples must have equal length for paired t-test")
        
        n = len(sample1)
        if n < 2:
            raise ValueError("Need at least 2 pairs for t-test")
        
        # Calculate differences
        differences = [s1 - s2 for s1, s2 in zip(sample1, sample2)]
        
        # Mean and std of differences
        mean_diff = self.mean(differences)
        std_diff = self.std_dev(differences)
        
        # Standard error
        se = std_diff / math.sqrt(n)
        
        # t-statistic
        if se == 0:
            t_stat = 0.0 if mean_diff == 0 else float('inf') * (1 if mean_diff > 0 else -1)
        else:
            t_stat = mean_diff / se
        
        # Degrees of freedom
        df = n - 1
        
        # p-value
        if alternative == 'two-sided':
            p_value = 2 * (1 - self._t_cdf(abs(t_stat), df))
        elif alternative == 'less':
            p_value = self._t_cdf(t_stat, df)
        elif alternative == 'greater':
            p_value = 1 - self._t_cdf(t_stat, df)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")
        
        # Ensure p-value is in valid range
        p_value = max(0, min(1, p_value))
        
        # 95% confidence interval for mean difference
        # Use t critical value approximation
        t_crit = 1.96 + 2.4 / df + 0.2 / (df * df)  # Approximate for 95%
        ci_low = mean_diff - t_crit * se
        ci_high = mean_diff + t_crit * se
        
        return TTestResult(
            t_statistic=t_stat,
            p_value=p_value,
            degrees_of_freedom=df,
            mean_difference=mean_diff,
            std_error=se,
            confidence_interval_95=(ci_low, ci_high)
        )
    
    def cohens_d(
        self, 
        sample1: List[float], 
        sample2: List[float],
        paired: bool = True
    ) -> EffectSize:
        """
        Calculate Cohen's d effect size.
        
        For paired samples, uses the standard deviation of differences.
        
        Interpretation (Cohen, 1988):
        - |d| < 0.2: negligible
        - 0.2 ≤ |d| < 0.5: small
        - 0.5 ≤ |d| < 0.8: medium
        - |d| ≥ 0.8: large
        
        Args:
            sample1: First sample
            sample2: Second sample
            paired: Whether samples are paired
            
        Returns:
            EffectSize with Cohen's d and interpretation
        """
        if paired:
            if len(sample1) != len(sample2):
                raise ValueError("Paired samples must have equal length")
            
            differences = [s1 - s2 for s1, s2 in zip(sample1, sample2)]
            mean_diff = self.mean(differences)
            std_diff = self.std_dev(differences)
            
            d = mean_diff / std_diff if std_diff > 0 else 0
        else:
            # Unpaired: use pooled standard deviation
            n1, n2 = len(sample1), len(sample2)
            mean1, mean2 = self.mean(sample1), self.mean(sample2)
            var1, var2 = self.variance(sample1), self.variance(sample2)
            
            # Pooled std dev
            pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
            pooled_std = math.sqrt(pooled_var)
            
            d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        # Interpretation
        abs_d = abs(d)
        if abs_d < 0.2:
            interpretation = "negligible"
        elif abs_d < 0.5:
            interpretation = "small"
        elif abs_d < 0.8:
            interpretation = "medium"
        else:
            interpretation = "large"
        
        return EffectSize(cohens_d=d, interpretation=interpretation)
    
    def analyze_all(
        self,
        paired_data: Dict[str, Tuple[List[float], List[float]]]
    ) -> Dict[str, Dict]:
        """
        Perform complete analysis on all metrics.
        
        Args:
            paired_data: Dict mapping metric name to (rr_values, mgrr_values)
            
        Returns:
            Dict with descriptive stats, t-test results, and effect sizes for each metric
        """
        results = {}
        
        for metric_name, (rr_data, mgrr_data) in paired_data.items():
            rr_stats = self.descriptive_stats(rr_data)
            mgrr_stats = self.descriptive_stats(mgrr_data)
            
            try:
                ttest = self.paired_ttest(rr_data, mgrr_data)
                effect = self.cohens_d(rr_data, mgrr_data)
            except (ValueError, ZeroDivisionError):
                ttest = None
                effect = None
            
            results[metric_name] = {
                'rr_stats': rr_stats,
                'mgrr_stats': mgrr_stats,
                'ttest': ttest,
                'effect_size': effect,
                'improvement': {
                    'absolute': rr_stats.mean - mgrr_stats.mean,
                    'percentage': (
                        (rr_stats.mean - mgrr_stats.mean) / rr_stats.mean * 100
                        if rr_stats.mean != 0 else 0
                    )
                }
            }
        
        return results
    
    def generate_report(
        self,
        analysis_results: Dict[str, Dict],
        title: str = "Statistical Analysis Report"
    ) -> str:
        """
        Generate a formatted text report.
        
        Args:
            analysis_results: Output from analyze_all()
            title: Report title
            
        Returns:
            Formatted text report
        """
        lines = [
            "=" * 80,
            title.center(80),
            "=" * 80,
            ""
        ]
        
        for metric, data in analysis_results.items():
            lines.extend([
                f"\n{'─' * 40}",
                f"Metric: {metric.upper()}",
                f"{'─' * 40}",
            ])
            
            # Descriptive stats
            lines.append("\n[Descriptive Statistics]")
            lines.append(f"  RR:    mean={data['rr_stats'].mean:.4f}, std={data['rr_stats'].std_dev:.4f}, median={data['rr_stats'].median:.4f}")
            lines.append(f"  MG-RR: mean={data['mgrr_stats'].mean:.4f}, std={data['mgrr_stats'].std_dev:.4f}, median={data['mgrr_stats'].median:.4f}")
            
            # Improvement
            imp = data['improvement']
            direction = "↓" if imp['absolute'] > 0 else "↑" if imp['absolute'] < 0 else "="
            lines.append(f"\n[Improvement]")
            lines.append(f"  MG-RR vs RR: {imp['absolute']:+.4f} ({imp['percentage']:+.2f}%) {direction}")
            
            # T-test
            if data['ttest']:
                t = data['ttest']
                sig_stars = "***" if t.p_value < 0.001 else "**" if t.p_value < 0.01 else "*" if t.p_value < 0.05 else ""
                lines.append(f"\n[Paired T-Test]")
                lines.append(f"  t({t.degrees_of_freedom}) = {t.t_statistic:.4f}")
                lines.append(f"  p-value = {t.p_value:.6f} {sig_stars}")
                lines.append(f"  95% CI: [{t.confidence_interval_95[0]:.4f}, {t.confidence_interval_95[1]:.4f}]")
            
            # Effect size
            if data['effect_size']:
                e = data['effect_size']
                lines.append(f"\n[Effect Size]")
                lines.append(f"  Cohen's d = {e.cohens_d:.4f} ({e.interpretation})")
        
        lines.extend([
            "",
            "=" * 80,
            "Note: * p<0.05, ** p<0.01, *** p<0.001",
            "=" * 80
        ])
        
        return "\n".join(lines)


if __name__ == "__main__":
    # Demo with synthetic data
    print("=== Statistical Analysis Demo ===\n")
    
    import random
    random.seed(42)
    
    # Simulate 100 paired observations
    n = 100
    rr_stutters = [random.randint(3, 8) for _ in range(n)]
    mgrr_stutters = [max(0, s - random.randint(1, 3)) for s in rr_stutters]  # MG-RR generally better
    
    rr_wt = [random.uniform(20, 40) for _ in range(n)]
    mgrr_wt = [w + random.uniform(-2, 5) for w in rr_wt]  # Slight increase for MG-RR
    
    # Create analysis
    analyzer = StatisticalAnalysis()
    
    paired_data = {
        'total_stutter': (rr_stutters, mgrr_stutters),
        'avg_waiting_time': (rr_wt, mgrr_wt)
    }
    
    results = analyzer.analyze_all(paired_data)
    report = analyzer.generate_report(results, "MG-RR vs RR Analysis (Demo)")
    print(report)
