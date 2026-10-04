"""What the program's phase score rewards: illustrative profiles through the
real _phase_score / _combine_research_scores functions."""
from common import core

def phase(R, DD, PF, T, windows):
    m = core.BacktestMetrics(trades=T, total_return_pct=R, max_drawdown_pct=DD, profit_factor=PF)
    return core._phase_score(m, windows, 19)

cases = {
 "A steady moderate":          (30, 10, 1.8, 60,  [5, 3, 2, 4, -2, 6, 1, -4]),
 "B big return, big risk":     (120, 45, 1.5, 60, [40, 25, -15, 30, 10, -12, 20, 5]),
 "C 20 trades, tiny DD":       (25, 3, 3.0, 20,   [6, 0, 5, -1, 4, 0, 6, 0]),
 "D one lucky window":         (40, 12, 1.6, 60,  [45, -1, -2, -1, 3, -6, -1, 0]),
 "E many small steady wins":   (15, 6, 1.4, 200,  [2, 1.5, 3, 1, -2, 2, 2.5, 1]),
 "F same as A, DD hidden in open trades (closed DD 4%)": (30, 4, 1.8, 60, [5, 3, 2, 4, -2, 6, 1, -4]),
}
print(f"{'profile':55s} {'phase score':>11s}")
for k, v in cases.items():
    print(f"{k:55s} {phase(*v):11.3f}")
# discontinuity of the combination at zero
for s, v in [(0.001, 1.0), (-0.001, 1.0), (0.5, 0.5), (0.5, 0.05)]:
    print(f"combine(selection={s}, validation={v}) = {core._combine_research_scores(s, v, 0.6, 0.6, 1, 1):.4f}")
