import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# Исходные данные (Таблица 1)
arr = np.array([
    0, 2, 0, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 2, 1, 0, 0, 1, 0, 0, 2,
    2, 0, 2, 1, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0, 0, 2, 1, 1, 2, 1, 0, 0, 0, 2,
    0, 2
])

n = len(arr)
xs = np.sort(arr)
vals, counts = np.unique(arr, return_counts=True)
rel_freqs = counts / n
cum_freqs = np.cumsum(counts) / n

# Параметры задания
al = 0.05
a = 0.00
b = 1.03
lam0 = 2.10

# ── a) Вариационный ряд ───────────────────────────────────────────
print("=" * 115)
print("a) ВАРИАЦИОННЫЙ РЯД")
print("=" * 115)
print(f"{'x_i':<10} {'n_i':<10} {'w_i':<30} {'F*(x)':<30}")
print("-" * 115)
for v, c, w, f in zip(vals, counts, rel_freqs, cum_freqs):
    print(f"{v:<10} {c:<10} {w:<30} {f:<30}")
print(f"\nN = {n}")

# ── b) Числовые характеристики ────────────────────────────────────
m = np.mean(arr)
s2 = np.var(arr, ddof=0)  # Смещенная дисперсия (знаменатель n)
s = np.sqrt(s2)
med = np.median(arr)

# Вычисление асимметрии и эксцесса через центральные моменты
asi = np.sum((vals - m) ** 3 * rel_freqs) / s ** 3
exc = np.sum((vals - m) ** 4 * rel_freqs) / s ** 4 - 3
prb = np.sum(rel_freqs[(vals >= a) & (vals <= b)])

print("\n" + "=" * 55)
print("b) ЧИСЛОВЫЕ ХАРАКТЕРИСТИКИ")
print("=" * 55)
print(f"(i)   Математическое ожидание : {m}")
print(f"(ii)  Дисперсия               : {s2}")
print(f"(iii) Медиана                 : {med}")
print(f"(iv)  Асимметрия              : {asi}")
print(f"(v)   Эксцесс                 : {exc}")
print(f"(vi)  P(X \u2208 [{a}, {b}])          : {prb}")

# ── c) ОМП и ММ для Пуассона ──────────────────────────────────────
lam_mle = m
lam_mm = m

print("\n" + "=" * 55)
print("c) ОЦЕНКИ ПАРАМЕТРА \u03bb (Пуассон)")
print("=" * 55)
print(f"   ОМП: \u03bb_mle = x\u0304 = {lam_mle},  смещение = 0")
print(f"   ММ:  \u03bb_mm  = x\u0304 = {lam_mm},  смещение = 0")

# ── d) Асимптотический ДИ для \u03bb + ДИ для дисперсии ────────────────
z = stats.norm.ppf(1 - al / 2)
ci_lo_lam = lam_mle - z * np.sqrt(lam_mle / n)
ci_hi_lam = lam_mle + z * np.sqrt(lam_mle / n)

# Исправленная дисперсия (ddof=1) строго для интервальной оценки
s2u = np.var(arr, ddof=1)
x2lo = stats.chi2.ppf(al / 2, n - 1)
x2hi = stats.chi2.ppf(1 - al / 2, n - 1)
ci_lo_var = (n - 1) * s2u / x2hi
ci_hi_var = (n - 1) * s2u / x2lo

print("\n" + "=" * 55)
print(f"d) ДОВЕРИТЕЛЬНЫЕ ИНТЕРВАЛЫ  (\u03b1 = {al})")
print("=" * 55)
print(f"   z_{{\u03b1/2}} = {z}")
print(f"   \u03c7\u00b2_{{\u03b1/2}} = {x2lo},  \u03c7\u00b2_{{1-\u03b1/2}} = {x2hi}")
print(f"\n   {'Par':<8} {'Lw':>30} {'Up':>30}")
print(f"   {'-' * 70}")
print(f"   {'Mean':<8} {ci_lo_lam:>30} {ci_hi_lam:>30}")
print(f"   {'Var':<8} {ci_lo_var:>30} {ci_hi_var:>30}")


# ── \u03c7\u00b2: Построение таблиц с объединением малочисленных групп ──────
def build_chi2(arr, lam, n):
    max_val = int(np.max(arr))
    lw_l, up_l, pr_l, obs_l = [], [], [], []
    for v in range(0, max_val):
        lw_l.append(v)
        up_l.append(v + 1)
        pr_l.append(stats.poisson.pmf(v, lam))
        obs_l.append(int(np.sum(arr == v)))

    # Последняя группа [max_val, +\u221e)
    lw_l.append(max_val)
    up_l.append(np.inf)
    pr_l.append(1 - stats.poisson.cdf(max_val - 1, lam))
    obs_l.append(int(np.sum(arr >= max_val)))

    # Объединение правого хвоста
    while len(pr_l) > 1 and pr_l[-1] * n < 5:
        pr_l[-2] += pr_l[-1]
        obs_l[-2] += obs_l[-1]
        up_l[-2] = up_l[-1]
        pr_l.pop()
        obs_l.pop()
        lw_l.pop()
        up_l.pop()

    # Объединение левого хвоста
    while len(pr_l) > 1 and pr_l[0] * n < 5:
        pr_l[1] += pr_l[0]
        obs_l[1] += obs_l[0]
        lw_l[1] = lw_l[0]
        pr_l.pop(0)
        obs_l.pop(0)
        lw_l.pop(0)
        up_l.pop(0)

    nu = np.array(obs_l, dtype=float)
    pr = np.array(pr_l)
    npr = pr * n
    res = (nu - npr) / np.sqrt(npr)
    res2 = res ** 2
    X2 = np.sum(res2)
    return lw_l, up_l, nu, npr, res, res2, X2


def print_chi2(lw_l, up_l, nu, npr, res, res2, X2, df, title):
    k = len(nu)
    xa1 = stats.chi2.ppf(1 - al, df)
    pval = stats.chi2.sf(X2, df)
    print(f"\n{'=' * 160}")
    print(f"  {title}")
    print(f"{'=' * 160}")
    print(f"{'i':>4} {'lw':>6} {'up':>6} {'nu_i':>8} {'np_i':>30} "
          f"{'res':>30} {'res^2':>30} {'np>=5':>6}")
    print("-" * 160)
    for i in range(k):
        lw_s = f"{int(lw_l[i])}" if not np.isinf(lw_l[i]) else "-\u221e"
        up_s = f"{int(up_l[i])}" if not np.isinf(up_l[i]) else "+\u221e"
        ok = "\u2713" if npr[i] >= 5 else "\u2717"
        print(f"{i + 1:>4} {lw_s:>6} {up_s:>6} {nu[i]:>8.0f} {npr[i]:>30} "
              f"{res[i]:>30} {res2[i]:>30} {ok:>6}")
    print("-" * 160)
    print(f"{'Итого':>18} {np.sum(nu):>8.0f} {np.sum(npr):>30} "
          f"{'':>30} {X2:>30}")
    print(f"\n   k = {k},  df = {df}")
    print(f"   X\u00b2 = {X2}")
    print(f"   \u0447\u00b2_\u043a\u0440 (\u0431={al}, df={df}) = {xa1}")
    print(f"   X\u00b2 > \u0447\u00b2_\u043a\u0440 : {X2 > xa1}  \u2192  "
          f"{'Отвергаем H0' if X2 > xa1 else 'Нет оснований отвергнуть H0'}")
    print(f"   p-value = {pval}")


# ── e) Простая гипотеза H0: Poisson(\u03bb0 = 2.10) ───────────────────
lw_e, up_e, nu_e, npr_e, res_e, res2_e, X2_e = build_chi2(arr, lam0, n)
print_chi2(lw_e, up_e, nu_e, npr_e, res_e, res2_e, X2_e,
           len(nu_e) - 1,
           f"e) \u03c7\u00b2  простая гипотеза H0: Poisson(\u03bb0 = {lam0})")

# ── f) Сложная гипотеза H0: Poisson(\u03bb_mle) ───────────────────────
lw_f, up_f, nu_f, npr_f, res_f, res2_f, X2_f = build_chi2(arr, lam_mle, n)
print_chi2(lw_f, up_f, nu_f, npr_f, res_f, res2_f, X2_f,
           len(nu_f) - 1 - 1,
           f"f) \u03c7\u00b2  сложная гипотеза H0: Poisson(\u03bb_mle = {lam_mle})")

# ── Графики ───────────────────────────────────────────────────────
bins = np.append(vals - 0.5, vals[-1] + 0.5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ЭФР
x_ecdf = np.concatenate([[-1], vals, [vals[-1] + 1]])
y_ecdf = np.concatenate([[0], cum_freqs, [1]])
ax1.hlines(y_ecdf[:-1], x_ecdf[:-1], x_ecdf[1:], colors='b', lw=2)
ax1.scatter(vals, np.concatenate([[0], cum_freqs[:-1]]), color='blue', zorder=4)
ax1.scatter(vals, cum_freqs, color='white', edgecolors='blue', zorder=4)
ax1.set_title('Эмпирическая функция распределения $F^*(x)$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$F^*(x)$')
ax1.grid(True, linestyle='--')
ax1.set_xlim(x_ecdf[0], x_ecdf[-1])
ax1.set_ylim(-0.1, 1.1)

# Гистограмма
ax2.hist(arr, bins=bins, edgecolor='black', color='skyblue', rwidth=0.8, density=False)
ax2.set_xticks(vals)
ax2.set_title('Гистограмма частот')
ax2.set_xlabel('Варианты ($x_i$)')
ax2.set_ylabel('Частота ($n_i$)')
ax2.grid(True, linestyle='--')

plt.tight_layout()
plt.show()
