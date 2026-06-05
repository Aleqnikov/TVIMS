import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import kstwobign
from scipy.optimize import minimize

arr = np.array([
    8.65, 4.55, 4.42, 4.76, 8.03, 5.67, 4.97, 4.42, -4.43, 9.87,
    1.84, 8.64, 8.05, -2.52, 10.74, 12.11, 8.14, 4.36, 0.60, 0.91,
    4.57, 1.14, -8.08, 0.38, 3.52, 4.07, -5.05, 6.05, 7.93, 6.75,
    3.97, -11.45, 2.83, 0.52, 2.30, 3.22, 2.03, 1.66, 0.92, 3.84,
    9.95, 2.32, 16.30, 6.47, 2.96, 5.93, 1.54, 10.51, -2.49, 5.86
])

al = 0.10
c = -1.00
d = 8.00
h = 2.00
a0 = -12.00
sig0 = 5.00

n = len(arr)
xs = np.sort(arr)

# ── a) Вариационный ряд ───────────────────────────────────────────
print("=" * 65)
print("a) ВАРИАЦИОННЫЙ РЯД")
print("=" * 65)
print(f"{'i':<6} {'x_(i)':<12} {'(i-1)/n':<12} {'i/n':<12}")
print("-" * 42)
for i, x in enumerate(xs):
    print(f"{i + 1:<6} {x:<12.4f} {i / n:<12.4f} {(i + 1) / n:<12.4f}")
print(f"\nN = {n}")

# ── b) Числовые характеристики ────────────────────────────────────
m = np.mean(arr)
s2 = np.mean((arr - m) ** 2)
s = np.sqrt(s2)

# Твой оригинальный расчет медианы
if n % 2 == 0:
    med = xs[n // 2 - 1]
else:
    med = xs[n // 2]

asi = np.mean((arr - m) ** 3) / s ** 3
exc = np.mean((arr - m) ** 4) / s ** 4 - 3
prb = np.mean((arr >= c) & (arr <= d))

print("\n" + "=" * 55)
print("b) ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ")
print("=" * 55)
print(f"(i)   Математическое ожидание : {m:.4f}")
print(f"(ii)  Дисперсия               : {s2:.4f}")
print(f"(iii) Медиана                 : {med:.4f}")
print(f"(iv)  Асимметрия              : {asi:.4f}")
print(f"(v)   Эксцесс                 : {exc:.4f}")
print(f"(vi)  P(X ∈ [{c}, {d}])       : {prb:.4f}")

# ── c) ОМП и ММ для нормального распределения ─────────────────────
a_mle = m
s2_mle = s2
bias_a = 0.0
bias_s2 = -s2 / n

print("\n" + "=" * 55)
print("c) ОМП И ММ ДЛЯ НОРМАЛЬНОГО РАСПРЕДЕЛЕНИЯ")
print("=" * 55)
print(f"   ОМП/ММ: a_mle  = x̄  = {a_mle:.4f},  смещение = {bias_a}")
print(f"   ОМП/ММ: s²_mle = s² = {s2_mle:.4f},  смещение ≈ {bias_s2:.4f}")

# ── d) Доверительные интервалы ────────────────────────────────────
s2_unb = np.var(arr, ddof=1)
s_unb = np.sqrt(s2_unb)
t_crit = stats.t.ppf(1 - al / 2, df=n - 1)
ci_a_lo = m - t_crit * s_unb / np.sqrt(n)
ci_a_hi = m + t_crit * s_unb / np.sqrt(n)
x2lo = stats.chi2.ppf(al / 2, df=n - 1)
x2hi = stats.chi2.ppf(1 - al / 2, df=n - 1)
ci_s2_lo = (n - 1) * s2_unb / x2hi
ci_s2_hi = (n - 1) * s2_unb / x2lo

print("\n" + "=" * 55)
print(f"d) ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ  (α = {al})")
print("=" * 55)
print(f"   t_{{α/2, n-1}} = {t_crit:.4f}")
print(f"   χ²_{{α/2}}     = {x2lo:.4f},  χ²_{{1-α/2}} = {x2hi:.4f}")
print(f"\n   {'Par':<8} {'Lw':>12} {'Up':>12}")
print(f"   {'-' * 34}")
print(f"   {'Mean':<8} {ci_a_lo:>12.4f} {ci_a_hi:>12.4f}")
print(f"   {'Var':<8} {ci_s2_lo:>12.4f} {ci_s2_hi:>12.4f}")

# ── e) Критерий Колмогорова ───────────────────────────────────────
ks_L = np.arange(0, n) / n
ks_R = np.arange(1, n + 1) / n
ks_F0 = stats.norm.cdf(xs, loc=a0, scale=sig0)

ks_c6 = np.abs(ks_F0 - ks_L)
ks_c7 = np.abs(ks_F0 - ks_R)
ks_mx = np.maximum(ks_c6, ks_c7)

j_ks = np.argmax(ks_mx)
D = ks_mx[j_ks]
D_n = D * np.sqrt(n)
ks_crit = kstwobign.ppf(1 - al)
ks_pval = kstwobign.sf(D_n)  # Соответствует ks.test() из R кода препода

print("\n" + "=" * 90)
print(f"e) ТАБЛИЦА КОЛМОГОРОВА  H0: N(a0={a0}, σ0={sig0})")
print("=" * 90)
print(f"{'i':>3} | {'x_i':>8} | {'(i-1)/n':>8} | {'i/n':>8} | {'F0(x_i)':>8} | {'|3-5|':>8} | {'|4-5|':>8} | {'max':>8}")
print("-" * 90)

for i in range(n):
    mark = "<-- MAX" if i == j_ks else ""
    print(f"{i+1:3} | {xs[i]:8.4f} | {ks_L[i]:8.4f} | {ks_R[i]:8.4f} | "
          f"{ks_F0[i]:8.4f} | {ks_c6[i]:8.4f} | {ks_c7[i]:8.4f} | {ks_mx[i]:8.4f} {mark}")

print("-" * 90)
print(f"   D = {D:.4f},  D*sqrt(n) = {D_n:.4f}")
print(f"   Критическое значение: {ks_crit:.4f}")
print(f"   p-value (наибольший уровень значимости): {ks_pval:.4e}")
print(f"   Итог: {'Отвергаем H0' if D_n > ks_crit else 'Нет оснований отвергнуть H0'}")


# ── СЕТКА ИНТЕРВАЛОВ И ЧАСТОТ ИЗ ГИСТОГРАММЫ (Пункты f и g) ───────
# Проектируем 5 объединенных интервалов на базе шага h=2
chi2_edges = [-np.inf, 0.0, 2.0, 4.0, 6.0, np.inf]
nu_observed = np.array([6, 9, 9, 11, 15], dtype=float)

# ── f) Простая гипотеза χ²: N(a0, sig0²) ─────────────────────────
k = len(nu_observed)
npr_f = []
for i in range(k):
    p = stats.norm.cdf(chi2_edges[i+1], a0, sig0) - stats.norm.cdf(chi2_edges[i], a0, sig0)
    npr_f.append(p * n)
npr_f = np.array(npr_f, dtype=float)

safe_npr_f = np.where(npr_f > 1e-10, npr_f, 1e-10)
res2_f = ((nu_observed - npr_f) ** 2) / safe_npr_f
X2_f = np.sum(res2_f)
df_f = k - 1
xa1_f = stats.chi2.ppf(1 - al, df_f)
pval_f = stats.chi2.sf(X2_f, df_f)

print(f"\n{'=' * 76}\n  f) χ²  простая гипотеза H0: N(a0={a0}, σ0={sig0})\n{'=' * 76}")
print(f"{'i':>4} {'lw':>8} {'up':>8} {'nu_i':>8} {'np_i':>8} {'(nu-np)^2/np':>14} {'np>=5':>6}")
print("-" * 76)
for i in range(k):
    lw_s = f"{chi2_edges[i]:.2f}" if not np.isinf(chi2_edges[i]) else "-∞"
    up_s = f"{chi2_edges[i+1]:.2f}" if not np.isinf(chi2_edges[i+1]) else "+∞"
    ok = "✓" if npr_f[i] >= 4.99 else "✗"
    print(f"{i + 1:>4} {lw_s:>8} {up_s:>8} {nu_observed[i]:>8.0f} {npr_f[i]:>8.3f} {res2_f[i]:>14.4f} {ok:>6}")
print("-" * 76)
print(f"{'Итого':>22} {n:>8.0f} {np.sum(npr_f):>8.3f} {X2_f:>14.4f}")
print(f"\n   k = {k},  df = {df_f}")
print(f"   X² = {X2_f:.4f}")
print(f"   χ²_кр (α={al}, df={df_f}) = {xa1_f:.4f}")
print(f"   X² > χ²_кр : {X2_f > xa1_f}  →  {'Отвергаем H0' if X2_f > xa1_f else 'Нет оснований отвергнуть H0'}")
print(f"   p-value = {pval_f:.4e}")


# ── g) Сложная гипотеза χ²: Поиск параметров через минимум Хи-Квадрат ──
# Реализация nlm() оптимизации из R-кода препода (блок #7)
def csq_target(params):
    a_opt, sig_opt = params
    if sig_opt <= 0:
        return 1e10
    p = stats.norm.cdf(chi2_edges[1:], loc=a_opt, scale=sig_opt) - stats.norm.cdf(chi2_edges[:-1], loc=a_opt, scale=sig_opt)
    npr = p * n
    safe_npr = np.where(npr > 1e-10, npr, 1e-10)
    return np.sum((nu_observed - safe_npr)**2 / safe_npr)

# Начальное приближение берем из точечных оценок (m, s)
opt_res = minimize(csq_target, [m, np.sqrt(s2)], method='Nelder-Mead')
a_opt, sig_opt = opt_res.x
X2_g = opt_res.fun

# Расчет финальных теоретических частот для таблицы g
p_g = stats.norm.cdf(chi2_edges[1:], loc=a_opt, scale=sig_opt) - stats.norm.cdf(chi2_edges[:-1], loc=a_opt, scale=sig_opt)
npr_g = p_g * n
res2_g = ((nu_observed - npr_g) ** 2) / npr_g
df_g = k - 1 - 2  # Оценивали 2 параметра через оптимизатор Хи-Квадрат, df уменьшается на 2
xa1_g = stats.chi2.ppf(1 - al, df_g)
pval_g = stats.chi2.sf(X2_g, df_g)

print(f"\n{'=' * 76}\n  g) χ²  сложная гипотеза H0: N(a_opt={a_opt:.4f}, σ_opt={sig_opt:.4f})\n{'=' * 76}")
print(f"{'i':>4} {'lw':>8} {'up':>8} {'nu_i':>8} {'np_i':>8} {'(nu-np)^2/np':>14} {'np>=5':>6}")
print("-" * 76)
for i in range(k):
    lw_s = f"{chi2_edges[i]:.2f}" if not np.isinf(chi2_edges[i]) else "-∞"
    up_s = f"{chi2_edges[i+1]:.2f}" if not np.isinf(chi2_edges[i+1]) else "+∞"
    ok = "✓" if npr_g[i] >= 4.99 else "✗"
    print(f"{i + 1:>4} {lw_s:>8} {up_s:>8} {nu_observed[i]:>8.0f} {npr_g[i]:>8.3f} {res2_g[i]:>14.4f} {ok:>6}")
print("-" * 76)
print(f"{'Итого':>22} {n:>8.0f} {np.sum(npr_g):>8.3f} {X2_g:>14.4f}")
print(f"\n   k = {k},  df = {df_g}")
print(f"   X² = {X2_g:.4f}")
print(f"   χ²_кр (α={al}, df={df_g}) = {xa1_g:.4f}")
print(f"   X² > χ²_кр : {X2_g > xa1_g}  →  {'Отвергаем H0' if X2_g > xa1_g else 'Нет оснований отвергнуть H0'}")
print(f"   p-value = {pval_g:.4e}")


# ── Графики ───────────────────────────────────────────────────────
x_min_plot = np.floor(np.min(arr) / h) * h
x_max_plot = np.ceil(np.max(arr) / h) * h
bins = np.arange(x_min_plot, x_max_plot + h / 2, h)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ЭФР
ecdf_x_l = np.concatenate([[xs[0] - 2], xs])
ecdf_x_r = np.concatenate([xs, [xs[-1] + 2]])
ecdf_y = np.arange(0, n + 1) / n

axes[0].hlines(ecdf_y, ecdf_x_l, ecdf_x_r, colors='b', lw=2)
axes[0].scatter(ecdf_x_r[:-1], ecdf_y[:-1], color='blue', zorder=4, s=20)
axes[0].scatter(ecdf_x_l[1:], ecdf_y[1:], color='white', edgecolors='blue', zorder=4, s=20)
axes[0].set_title('ЭФР $F^*(x)$')
axes[0].set_xlabel('$x$')
axes[0].set_ylabel('$F^*(x)$')
axes[0].grid(True, linestyle='--')

# Гистограмма + полигон
counts_h, edges = np.histogram(arr, bins=bins)
mids = (edges[:-1] + edges[1:]) / 2
axes[1].bar(edges[:-1], counts_h, width=h, align='edge',
            edgecolor='black', color='skyblue', alpha=0.7, label='Гистограмма')
axes[1].plot(mids, counts_h, 'ro-', lw=2, label='Полигон')
axes[1].set_title(f'Гистограмма и полигон (h={h})')
axes[1].set_xlabel('$x$')
axes[1].set_ylabel('Частота')
axes[1].legend()
axes[1].grid(True, linestyle='--')

# Гистограмма + плотности
axes[2].hist(arr, bins=bins, density=True, edgecolor='black',
             color='skyblue', alpha=0.7, label='Эмпирич.')

x_line = np.linspace(a0 - 3 * sig0, xs[-1] + 2, 400)
axes[2].plot(x_line, stats.norm.pdf(x_line, a_opt, sig_opt),
             'r-', lw=2, label=f'Сложная (Опт): N({a_opt:.2f}, {sig_opt**2:.2f})')
axes[2].plot(x_line, stats.norm.pdf(x_line, a0, sig0),
             'g--', lw=2, label=f'Простая: N({a0}, {sig0 ** 2:.0f})')
axes[2].set_title('Плотность распределения')
axes[2].set_xlabel('$x$')
axes[2].set_ylabel('Плотность')
axes[2].legend()
axes[2].grid(True, linestyle='--')

plt.tight_layout()
plt.show()
