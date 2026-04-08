
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
import matplotlib.patheffects as pe

# =============================================================================
# 1. Global style
# =============================================================================
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.0
plt.rcParams['xtick.major.width'] = 1.0
plt.rcParams['ytick.major.width'] = 1.0
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['axes.facecolor'] = '#FAFAFA'
plt.rcParams['figure.facecolor'] = '#FFFFFF'

COLORS = {
    'neg': '#A8E6CF',
    'neg_fill': '#E8FFF5',
    'weak': '#FFD3B6',
    'weak_fill': '#FFF1E5',
    'pos': '#FFB5C2',
    'pos_fill': '#FFE7ED',
    'accent': '#C7CEEA',
    'neg_edge': '#31A384',
    'weak_edge': '#F4A261',
    'pos_edge': '#E26A8D',
    'gray': '#8E8E93',
    'black': '#333333'
}

COVID_TERNARY_SPLIT_FEATURE = 'sqrt_response'
DEFAULT_SEED = 42
DEFAULT_N_SHUFFLES = 800

def fancy_line(ax, x, y, color, lw=2.0, **kwargs):
    line, = ax.plot(x, y, color=color, lw=lw, **kwargs)
    line.set_path_effects([
        pe.Stroke(linewidth=lw + 1.2, foreground='white'),
        pe.Normal()
    ])
    return line

def fancy_text(ax, x, y, s, **kwargs):
    txt = ax.text(x, y, s, **kwargs)
    txt.set_path_effects([
        pe.Stroke(linewidth=1.2, foreground='white'),
        pe.Normal()
    ])
    return txt

def add_panel_label(ax, label, y, fontsize=11.5):
    ax.text(
        0.5, y, f'({label})',
        transform=ax.transAxes,
        ha='center', va='top',
        fontsize=fontsize, fontweight='bold'
    )

# =============================================================================
# 2. Data loading
# =============================================================================

def zscore_normalize(values, neg_reference):
    mu = np.mean(neg_reference)
    sigma = np.std(neg_reference, ddof=1)
    if sigma < 1e-9:
        sigma = 1.0
    return (values - mu) / sigma, mu, sigma

def safe_balanced_accuracy(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(y_true)
    if len(labels) == 0:
        return np.nan
    recalls = [
        np.mean(y_pred[y_true == label] == label)
        for label in labels
    ]
    return float(np.mean(recalls))

def sample_truncated_normal(rng, loc, scale, low, high, size):
    out = np.empty(size, dtype=float)
    filled = 0
    while filled < size:
        draws = rng.normal(loc=loc, scale=scale, size=max(16, 4 * (size - filled)))
        draws = draws[(draws >= low) & (draws <= high)]
        if len(draws) == 0:
            continue
        take = min(len(draws), size - filled)
        out[filled:filled + take] = draws[:take]
        filled += take
    return out

def sample_truncated_lognormal(rng, mean_log, sigma_log, low, high, size):
    out = np.empty(size, dtype=float)
    filled = 0
    while filled < size:
        draws = rng.lognormal(mean=mean_log, sigma=sigma_log, size=max(16, 4 * (size - filled)))
        draws = draws[(draws >= low) & (draws <= high)]
        if len(draws) == 0:
            continue
        take = min(len(draws), size - filled)
        out[filled:filled + take] = draws[:take]
        filled += take
    return out

def generate_covid_surrogate_negatives(data_resp, n_neg=12, negative_model='uniform'):
    rng = np.random.RandomState(DEFAULT_SEED)
    clia_low, clia_high = 0.1, 0.9
    min_positive_response = float(np.min(data_resp))
    resp_low = 0.1 * min_positive_response
    resp_high = 0.8 * min_positive_response

    if negative_model == 'uniform':
        clia_neg = rng.uniform(clia_low, clia_high, n_neg)
        resp_neg = rng.uniform(resp_low, resp_high, n_neg)
        return clia_neg, resp_neg

    if negative_model == 'truncated_noise':
        # A weak latent interference term introduces correlated nuisance shifts
        # while keeping all surrogate negatives inside the stated assay range.
        interference = sample_truncated_normal(rng, loc=0.0, scale=0.75, low=-1.5, high=1.5, size=n_neg)
        clia_neg = np.array([
            sample_truncated_normal(
                rng,
                loc=0.35 + 0.08 * eta,
                scale=0.14,
                low=clia_low,
                high=clia_high,
                size=1,
            )[0]
            for eta in interference
        ], dtype=float)

        base_log = np.log(np.sqrt(resp_low * resp_high))
        resp_neg = np.array([
            sample_truncated_lognormal(
                rng,
                mean_log=base_log + 0.18 * eta,
                sigma_log=0.38,
                low=resp_low,
                high=resp_high,
                size=1,
            )[0]
            for eta in interference
        ], dtype=float)
        return clia_neg, resp_neg

    raise ValueError(f'Unsupported negative_model: {negative_model}')

def load_hbv_data():
    labels = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 0])
    conc_A = np.array([1.19e-5, 5.18e-5, 1.07e-2, 4.96e-6, 7.89e-3,
                       8.09e-4, 2.88e-5, 2.23e-5, 3.93e-3, 5.15e-5])
    conc_B = np.array([3.29e-5, 3.83e-5, 4.94e-3, 1.15e-5, 1.90e-2,
                       2.30e-4, 1.85e-5, 4.69e-5, 2.71e-2, 5.09e-6])

    df = pd.DataFrame({
        'subject_id': list(range(1, 11)) + list(range(11, 21)),
        'group': np.repeat(['A', 'B'], 10),
        'conc': np.concatenate([conc_A, conc_B]),
        'label': np.concatenate([labels, labels])
    })
    df['log_val'] = np.log10(df['conc'] + 1e-12)
    return df

def load_covid_data(ternary_split_feature='log_val', negative_model='uniform'):
    data_pos = np.concatenate([
        [9.197, 9.104, 273.59, 8.747, 80.101, 11.603, 1.795, 8.548,
         384.06, 4.802, 33.070, 4.826, 15.84, 429.36, 3.629],
        [9.197, 9.104, 273.59, 8.747, 80.101, 11.603, 1.795, 8.548,
         384.06, 4.802, 33.070, 4.826, 15.84, 429.36, 3.629]
    ])
    data_resp = np.concatenate([
        [3.46e-4, 3.34e-5, 1.74e-1, 1.72e-4, 2.27e-3, 1.81e-4, 2.26e-5,
         2.16e-4, 2.04e-2, 2.75e-5, 1.38e-3, 1.15e-4, 6.95e-4, 4.82e-2,
         2.98e-5],
        [2.76e-4, 3.74e-5, 1.50e-2, 5.99e-5, 2.00e-3, 2.67e-4, 7.25e-6,
         1.51e-4, 3.67e-1, 3.13e-5, 4.00e-3, 5.99e-5, 1.15e-4, 1.12e-2,
         1.77e-5]
    ])

    n_neg = 12
    clia_neg, resp_neg = generate_covid_surrogate_negatives(
        data_resp,
        n_neg=n_neg,
        negative_model=negative_model,
    )

    parts_clia = [clia_neg, data_pos]
    parts_resp = [resp_neg, data_resp]

    df = pd.DataFrame({
        'CLIA': np.concatenate(parts_clia),
        'response': np.concatenate(parts_resp)
    })
    df['log_val'] = np.log10(df['response'] + 1e-12)
    df['label_binary'] = (df['CLIA'] >= 1.0).astype(int)

    subject_ids = list(range(100, 112))
    subject_ids += list(range(1, 16))
    subject_ids += list(range(16, 31))
    df['subject_id'] = subject_ids

    df = assign_covid_ternary_labels(df, split_feature=ternary_split_feature)
    return df

def assign_covid_ternary_labels(df, split_feature='log_val', min_component_samples=1):
    df = df.copy()
    pos_mask = df['label_binary'] == 1
    if split_feature == 'log_val':
        split_values = df['log_val'].values.astype(float)
    elif split_feature == 'response':
        split_values = df['response'].values.astype(float)
    elif split_feature == 'sqrt_response':
        split_values = np.sqrt(df['response'].values.astype(float))
    elif split_feature == 'CLIA':
        split_values = df['CLIA'].values.astype(float)
    else:
        raise ValueError(f'Unsupported split_feature: {split_feature}')

    X_pos = split_values[pos_mask].reshape(-1, 1)

    gmm1 = GaussianMixture(n_components=1, random_state=DEFAULT_SEED).fit(X_pos)
    gmm2 = GaussianMixture(n_components=2, random_state=DEFAULT_SEED).fit(X_pos)
    bic1 = gmm1.bic(X_pos)
    bic2 = gmm2.bic(X_pos)
    delta_bic = bic1 - bic2

    labels = df['label_binary'].values.astype(int)
    if delta_bic > 0:
        means = gmm2.means_.flatten()
        order = np.argsort(means)
        resp = gmm2.predict_proba(X_pos)
        comps = resp.argmax(axis=1)
        comp_counts = np.bincount(comps, minlength=2)
        if np.min(comp_counts) >= min_component_samples:
            mapping = {order[0]: 1, order[1]: 2}
            labels[pos_mask] = [mapping[c] for c in comps]
        else:
            labels[pos_mask] = 2
    else:
        labels[pos_mask] = 2

    df['label_ternary'] = labels
    df['bic_k1'] = bic1
    df['bic_k2'] = bic2
    df['delta_bic'] = delta_bic
    df['ternary_split_feature'] = split_feature
    return df



# =============================================================================
# 3. Softmax tools and full-data prototypes for main figures only
# =============================================================================

def softmax_binary(z, p0, p1, T=1.0):
    scale = max(T, 1e-8)
    d0 = (z - p0) ** 2 / (2 * scale ** 2)
    d1 = (z - p1) ** 2 / (2 * scale ** 2)
    e0, e1 = np.exp(-d0), np.exp(-d1)
    return e1 / (e0 + e1 + 1e-12)

def softmax_cond(z, p_w, p_s, T=1.0):
    scale = max(T, 1e-8)
    dw = (z - p_w) ** 2 / (2 * scale ** 2)
    ds = (z - p_s) ** 2 / (2 * scale ** 2)
    ew, es = np.exp(-dw), np.exp(-ds)
    return ew / (ew + es + 1e-12), es / (ew + es + 1e-12)

hbv_df = load_hbv_data()
hbv_neg = hbv_df[hbv_df['label'] == 0]['log_val'].values
hbv_df['z'], hbv_mu_neg, hbv_sigma_neg = zscore_normalize(hbv_df['log_val'].values, hbv_neg)
hbv_p_neg = hbv_df[hbv_df['label'] == 0]['z'].mean()
hbv_p_pos = hbv_df[hbv_df['label'] == 1]['z'].mean()
HBV_SCALE = (hbv_p_pos - hbv_p_neg) / 3.0

covid_df = load_covid_data(
    ternary_split_feature=COVID_TERNARY_SPLIT_FEATURE,
)
covid_neg = covid_df[covid_df['label_ternary'] == 0]['log_val'].values
covid_df['z'], covid_mu_neg, covid_sigma_neg = zscore_normalize(covid_df['log_val'].values, covid_neg)
covid_p_neg = covid_df[covid_df['label_ternary'] == 0]['z'].mean()
covid_p_pos_g = covid_df[covid_df['label_ternary'] >= 1]['z'].mean()


def _proto_mean(df, label, fallback):
    vals = df[df['label_ternary'] == label]['z'].values
    return vals.mean() if len(vals) > 0 else fallback


def _local_band_offsets(z_vals, collision_threshold=0.55):
    z_vals = np.asarray(z_vals, dtype=float)
    n = len(z_vals)
    if n == 0:
        return np.array([], dtype=float)

    order = np.argsort(z_vals)
    sorted_vals = z_vals[order]
    offsets_sorted = np.zeros(n, dtype=float)

    start = 0
    while start < n:
        end = start + 1
        while end < n and (sorted_vals[end] - sorted_vals[end - 1]) < collision_threshold:
            end += 1
        cluster_size = end - start
        if cluster_size > 1:
            spread = [0.0]
            step = 1
            while len(spread) < cluster_size:
                spread.append(float(step))
                if len(spread) < cluster_size:
                    spread.append(float(-step))
                step += 1
            offsets_sorted[start:end] = np.asarray(spread[:cluster_size], dtype=float)
        start = end

    offsets = np.zeros(n, dtype=float)
    offsets[order] = offsets_sorted
    return offsets


covid_p_weak = _proto_mean(covid_df, 1, covid_p_pos_g)
covid_p_strong = _proto_mean(covid_df, 2, covid_p_pos_g if not np.isnan(covid_p_pos_g) else covid_p_neg)
COVID_L1_SCALE = (covid_p_pos_g - covid_p_neg) / 3.0
COVID_L2_SCALE = (covid_p_strong - covid_p_weak) / 3.0


# =============================================================================
# 4. Main figures
# =============================================================================

def plot_fig_j_hbv(output='Fig5J_HBV_Macaron.png'):
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    z_min = hbv_df['z'].min() - 1
    z_max = hbv_df['z'].max() + 1
    z = np.linspace(z_min, z_max, 400)
    cutoff = (hbv_p_neg + hbv_p_pos) / 2

    prob_pos = softmax_binary(z, hbv_p_neg, hbv_p_pos, T=HBV_SCALE)

    ax.axvspan(z_min, cutoff, color=COLORS['neg_fill'], alpha=0.9, zorder=0)
    ax.axvspan(cutoff, z_max, color=COLORS['pos_fill'], alpha=0.9, zorder=0)

    ax.fill_between(z, 0, 1 - prob_pos, color=COLORS['neg'], alpha=0.12, zorder=1)
    ax.fill_between(z, 0, prob_pos, color=COLORS['pos'], alpha=0.12, zorder=1)

    fancy_line(ax, z, 1 - prob_pos, color=COLORS['neg_edge'], lw=2.0,
               label='P(Negative)', zorder=3)
    fancy_line(ax, z, prob_pos, color=COLORS['pos_edge'], lw=2.0,
               label='P(Positive)', zorder=3)

    bins = np.linspace(z_min, z_max, 8)
    ax.hist(hbv_df[hbv_df['label'] == 0]['z'], bins=bins, density=True,
            color=COLORS['neg'], alpha=0.35, edgecolor='none',
            label='Gold-standard\nnegatives', zorder=2)
    ax.hist(hbv_df[hbv_df['label'] == 1]['z'], bins=bins, density=True,
            color=COLORS['pos'], alpha=0.35, edgecolor='none',
            label='CLIA positives', zorder=2)

    ax.axvline(hbv_p_neg, color=COLORS['neg_edge'], lw=1.4, linestyle=':')
    ax.axvline(hbv_p_pos, color=COLORS['pos_edge'], lw=1.4, linestyle=':')
    ax.axvline(cutoff, color=COLORS['gray'], lw=1.2, linestyle='--')

    ymax = ax.get_ylim()[1]
    ax.scatter([hbv_p_neg, hbv_p_pos], [ymax * 0.96, ymax * 0.96],
               s=26, facecolor='white',
               edgecolor=[COLORS['neg_edge'], COLORS['pos_edge']],
               lw=1.2, zorder=4)

    fancy_text(ax, hbv_p_neg, ymax * 0.92, 'C$_{neg}$',
               color=COLORS['neg_edge'], ha='left', va='top', fontsize=9)
    fancy_text(ax, hbv_p_pos, ymax * 0.92, 'C$_{pos}$',
               color=COLORS['pos_edge'], ha='left', va='top', fontsize=9)
    ax.text(cutoff, ymax * 0.95, 'Cutoff', color=COLORS['gray'],
            ha='left', va='bottom', fontsize=9)

    ax.set_xlim(z_min, z_max)
    ax.set_xlabel('Standardized signal z (HBsAb, sweat)', fontweight='bold')
    ax.set_ylabel('Density / P(class)', fontweight='bold')
    ax.legend(frameon=True, fontsize=8, loc='upper right',
              bbox_to_anchor=(0.98, 0.92), markerscale=0.8,
              facecolor='white', edgecolor='#DDDDDD', framealpha=0.95)
    fig.savefig(output, dpi=600)

def plot_fig_k_covid(output='Fig5K_COVID_Macaron.png'):
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    z_all = covid_df['z'].values
    z_min = z_all.min() - 1.5
    z_max = z_all.max() + 1.5

    class_info = [
        (0, 'Negative', COLORS['neg'], COLORS['neg_edge']),
        (1, 'Weak positive', COLORS['weak'], COLORS['weak_edge']),
        (2, 'Strong positive', COLORS['pos'], COLORS['pos_edge'])
    ]

    for y, (lab, name, c_fill, c_edge) in enumerate(class_info):
        vals = covid_df[covid_df['label_ternary'] == lab]['z'].values
        if len(vals) == 0:
            continue
        ax.axhspan(y - 0.32, y + 0.32, color=c_fill, alpha=0.10, zorder=0)
        rng = np.random.RandomState(DEFAULT_SEED + lab)
        jitter = (rng.rand(len(vals)) - 0.5) * 0.20
        ax.scatter(vals, np.full_like(vals, y) + jitter,
                   s=40, facecolor=c_fill, edgecolor=c_edge,
                   lw=0.8, alpha=0.95, zorder=3)
        mu = vals.mean()
        sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        ax.hlines(y, mu - sd, mu + sd, colors=c_edge, lw=3.0, zorder=4)
        ax.scatter(mu, y, s=36, facecolor='white', edgecolor=c_edge,
                   lw=1.2, zorder=5)

    ax.set_xlim(z_min, z_max)
    ax.set_ylim(-0.7, 2.3)
    ax.set_xlabel('Standardized signal z (NAb, sweat)', fontweight='bold')
    ax.set_yticks([0, 1, 2])
    ax.set_ylabel('Class', fontweight='bold')
    ax.set_position([0.18, 0.18, 0.78, 0.74])

    legend_elements = [Patch(facecolor=c_fill, edgecolor=c_edge, alpha=0.6, label=name)
                       for _, name, c_fill, c_edge in class_info]
    ax.legend(handles=legend_elements, fontsize=8, loc='lower right',
              bbox_to_anchor=(0.98, 0.05), frameon=True,
              facecolor='white', edgecolor='#DDDDDD', framealpha=0.95)
    fig.savefig(output, dpi=600)

def plot_fig_o_stacked(output='Fig5O_Stacked_Macaron.png'):
    fig, ax = plt.subplots(figsize=(7.2, 4.3))

    z_min = covid_df['z'].min() - 1.5
    z_max = covid_df['z'].max() + 1.5
    z = np.linspace(z_min, z_max, 500)

    p_pos_global = softmax_binary(z, covid_p_neg, covid_p_pos_g, T=COVID_L1_SCALE)
    pw_cond, ps_cond = softmax_cond(z, covid_p_weak, covid_p_strong, T=COVID_L2_SCALE)
    p_neg = 1 - p_pos_global
    p_weak = p_pos_global * pw_cond
    p_strong = p_pos_global * ps_cond

    stack = ax.stackplot(z, p_neg, p_weak, p_strong,
                         colors=[COLORS['neg_fill'], COLORS['weak_fill'], COLORS['pos_fill']],
                         alpha=1)

    ax.plot(z, p_neg, color='white', lw=1.0)
    ax.plot(z, p_neg + p_weak, color='white', lw=1.0)
    fancy_line(ax, z, p_neg, color=COLORS['neg_edge'], lw=1.2)
    fancy_line(ax, z, p_weak, color=COLORS['weak_edge'], lw=1.2)
    fancy_line(ax, z, p_strong, color=COLORS['pos_edge'], lw=1.2)

    cut_l1 = (covid_p_neg + covid_p_pos_g) / 2.0
    cut_l2 = (covid_p_weak + covid_p_strong) / 2.0
    ax.axvline(cut_l1, color=COLORS['gray'], lw=1.1, linestyle='--')
    ax.axvline(cut_l2, color=COLORS['gray'], lw=1.1, linestyle=':')
    ax.text(cut_l1, 1.02, 'Layer 1 cutoff', ha='left', va='bottom', fontsize=8, color=COLORS['gray'])
    ax.text(cut_l2, 1.02, 'Layer 2 cutoff', ha='left', va='bottom', fontsize=8, color=COLORS['gray'])

    x_neg = (z_min + cut_l1) / 2.0
    x_weak = (cut_l1 + cut_l2) / 2.0
    x_strong = (cut_l2 + z_max) / 2.0
    ax.text(x_neg, 0.09, 'Negative', color=COLORS['neg_edge'], ha='left', va='center', fontsize=8)
    ax.text(x_weak, 0.09, 'Weak positive', color=COLORS['weak_edge'], ha='left', va='center', fontsize=8)
    ax.text(x_strong, 0.09, 'Strong positive', color=COLORS['pos_edge'], ha='left', va='center', fontsize=8)

    class_offsets = {}
    for lbl in [0, 1, 2]:
        idxs = covid_df.index[covid_df['label_ternary'] == lbl].tolist()
        z_vals = covid_df.loc[idxs, 'z'].values.astype(float)
        class_offsets.update({
            idx: _local_band_offsets(z_vals)[i] for i, idx in enumerate(idxs)
        })

    for idx, row in covid_df.iterrows():
        val = row['z']
        lbl = row['label_ternary']
        p_pos_val = softmax_binary(val, covid_p_neg, covid_p_pos_g, T=COVID_L1_SCALE)
        pn = 1 - p_pos_val
        pw_c, ps_c = softmax_cond(val, covid_p_weak, covid_p_strong, T=COVID_L2_SCALE)
        pw_val = p_pos_val * pw_c
        ps_val = p_pos_val * ps_c

        if lbl == 0:
            low, high = 0.0, pn
            marker = 'o'
            color_fill = COLORS['neg']
        elif lbl == 1:
            low, high = pn, pn + pw_val
            marker = '^'
            color_fill = COLORS['weak']
        else:
            low, high = pn + pw_val, pn + pw_val + ps_val
            marker = 'D'
            color_fill = COLORS['pos']

        mid = 0.5 * (low + high)
        band_half = max(0.5 * (high - low), 1e-6)
        max_shift = min(0.025, 0.30 * band_half)
        step = min(0.008, 0.10 * band_half)
        y = mid + class_offsets.get(idx, 0.0) * step
        y = min(max(y, low + max_shift * 0.35), high - max_shift * 0.35)
        ax.scatter(val, y, facecolor=color_fill, edgecolor='white', marker=marker, s=36, lw=1.0, zorder=4)
        ax.scatter(val, y, facecolor='none', edgecolor=COLORS['black'], marker=marker, s=36, lw=0.7, zorder=5)

    ax.set_xlim(z_min, z_max)
    ax.set_ylim(0, 1.00)
    ax.set_xlabel('Standardized signal z (NAb, sweat)', fontweight='bold')
    ax.set_ylabel('Class probability', fontweight='bold')

    handles = [stack[0], stack[1], stack[2]]
    labels = ['P(Negative)', 'P(Weak positive)', 'P(Strong positive)']
    ax.legend(handles, labels, frameon=True, fontsize=8, loc='upper right',
              bbox_to_anchor=(0.98, 0.94), facecolor='white', edgecolor='#DDDDDD',
              framealpha=0.95)
    fig.savefig(output, dpi=600)

def plot_fig_m_bic(output='Fig5M_BIC_Macaron.png'):
    """Plot the positive-sample BIC comparison."""
    fig, ax = plt.subplots(figsize=(6.2, 4.2))

    pos_df = covid_df[covid_df['label_binary'] == 1].copy()
    pos_z = pos_df['z'].values.astype(float)
    weak_z = pos_df[pos_df['label_ternary'] == 1]['z'].values.astype(float)
    strong_z = pos_df[pos_df['label_ternary'] == 2]['z'].values.astype(float)
    z_grid = np.linspace(pos_z.min() - 1.5, pos_z.max() + 1.5, 500)

    split_samples = get_covid_split_values(pos_df, split_feature=COVID_TERNARY_SPLIT_FEATURE)
    split_X = split_samples.reshape(-1, 1)
    gmm1 = GaussianMixture(n_components=1, random_state=DEFAULT_SEED).fit(split_X)
    gmm2 = GaussianMixture(n_components=2, random_state=DEFAULT_SEED).fit(split_X)
    bic1 = float(gmm1.bic(split_X))
    bic2 = float(gmm2.bic(split_X))
    delta_bic = bic1 - bic2

    mu1 = float(np.mean(pos_z))
    std1 = float(np.std(pos_z, ddof=1))
    if std1 < 1e-9:
        std1 = 1.0

    means2 = np.array([np.mean(weak_z), np.mean(strong_z)], dtype=float)
    stds2 = np.array([np.std(weak_z, ddof=1), np.std(strong_z, ddof=1)], dtype=float)
    stds2 = np.where(stds2 < 1e-9, 1.0, stds2)
    weights = np.array([len(weak_z), len(strong_z)], dtype=float)
    weights = weights / weights.sum()

    for m, s, c_fill in zip(means2, stds2, [COLORS['weak_fill'], COLORS['pos_fill']]):
        ax.axvspan(m - s, m + s, color=c_fill, alpha=0.65, zorder=0)

    bins = np.linspace(z_grid.min(), z_grid.max(), 12)
    ax.hist(pos_z, bins=bins, density=True, color=COLORS['accent'], alpha=0.20,
            edgecolor='#E0E0E0', label='Positive samples', zorder=1)

    fancy_line(ax, z_grid, norm.pdf(z_grid, mu1, std1),
               color=COLORS['neg_edge'], lw=2.2, label='K=1 Gaussian', zorder=3)

    pdf1 = weights[0] * norm.pdf(z_grid, means2[0], stds2[0])
    pdf2 = weights[1] * norm.pdf(z_grid, means2[1], stds2[1])
    mixture = pdf1 + pdf2

    fancy_line(ax, z_grid, pdf1, color=COLORS['weak_edge'], lw=2.0,
               linestyle='--', label='K=2 comp. 1', zorder=4)
    fancy_line(ax, z_grid, pdf2, color=COLORS['pos_edge'], lw=2.0,
               linestyle='--', label='K=2 comp. 2', zorder=4)
    fancy_line(ax, z_grid, mixture, color=COLORS['black'], lw=2.3,
               linestyle='-', label='K=2 mixture', zorder=5)

    peak_y1 = weights[0] * norm.pdf(means2[0], means2[0], stds2[0])
    peak_y2 = weights[1] * norm.pdf(means2[1], means2[1], stds2[1])
    ax.scatter([means2[0], means2[1]], [peak_y1, peak_y2],
               s=36, facecolor='white', edgecolor=[COLORS['weak_edge'], COLORS['pos_edge']],
               lw=1.3, zorder=6)

    y_top = max(np.max(mixture), np.max(norm.pdf(z_grid, mu1, std1)), peak_y1, peak_y2)
    ax.set_ylim(0, y_top * 1.28)
    label_y = ax.get_ylim()[1] * 0.82
    label_specs = [
        (means2[0], means2[0] + 0.30 * stds2[0], COLORS['weak_edge'], 'Weak mode', 'left'),
        (means2[1], means2[1] - 0.85 * stds2[1], COLORS['pos_edge'], 'Strong mode', 'left'),
    ]
    for line_x, text_x, c, label, ha in label_specs:
        ax.axvline(line_x, color=c, lw=1.2, linestyle=':')
        ax.text(text_x, label_y, label,
                color=c, ha=ha, va='bottom', fontsize=8, zorder=7)

    prefix1, prefix2, prefix3 = 'BIC(K=1)', 'BIC(K=2)', 'Delta BIC'
    pad = max(len(prefix1), len(prefix2), len(prefix3))
    txt = f'{prefix1:<{pad}} = {bic1:.1f}\n{prefix2:<{pad}} = {bic2:.1f}\n{prefix3:<{pad}} = {delta_bic:.1f}'
    ax.set_xlim(z_grid.min(), z_grid.max())
    ax.set_xlabel('Standardized signal z (NAb, sweat, positives)', fontweight='bold')
    ax.set_ylabel('Density', fontweight='bold')

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bic_text = ax.text(0.74, 0.98, txt, transform=ax.transAxes,
                       ha='left', va='top', fontsize=8, fontfamily='monospace')
    leg = ax.legend(frameon=False, fontsize=8, loc='upper left',
                    bbox_to_anchor=(0.73, 0.85), markerscale=0.8,
                    borderaxespad=0.0, labelspacing=0.35,
                    handlelength=2.6, handletextpad=0.6)
    box_bic = bic_text.get_window_extent(renderer=renderer)
    box_leg = leg.get_window_extent(renderer=renderer)
    from matplotlib.transforms import Bbox
    merged = Bbox.union([box_bic, box_leg]).expanded(1.05, 1.05)
    merged_inv = merged.transformed(ax.transAxes.inverted())
    ax.add_patch(plt.Rectangle((merged_inv.xmin, merged_inv.ymin), merged_inv.width, merged_inv.height,
                               transform=ax.transAxes, facecolor='white', edgecolor='#DDDDDD',
                               alpha=0.95, zorder=0))
    bic_text.set_zorder(3)
    leg.set_zorder(3)
    fig.savefig(output, dpi=600)

# =============================================================================
# 5. Supplementary figures
# =============================================================================

def fit_binary_proto_model(train_log, train_y):
    train_y = np.asarray(train_y).astype(int)
    train_log = np.asarray(train_log, dtype=float)
    neg_log = train_log[train_y == 0]
    if len(neg_log) >= 2:
        mu = float(np.mean(neg_log))
        sigma = float(np.std(neg_log, ddof=1))
    else:
        mu = float(np.mean(train_log))
        sigma = float(np.std(train_log, ddof=1)) if len(train_log) > 1 else 1.0
    if sigma < 1e-9:
        sigma = 1.0

    z_train = (train_log - mu) / sigma
    z_neg = z_train[train_y == 0]
    z_pos = z_train[train_y == 1]
    c_neg = float(np.mean(z_neg)) if len(z_neg) else float(np.mean(z_train))
    c_pos = float(np.mean(z_pos)) if len(z_pos) else c_neg + 1.0
    T = abs((c_pos - c_neg) / 3.0)
    if T < 1e-6:
        T = 1.0
    return {'mu': mu, 'sigma': sigma, 'c_neg': c_neg, 'c_pos': c_pos, 'T': T}

def predict_binary_proto(model, log_values):
    z = (np.asarray(log_values, dtype=float) - model['mu']) / model['sigma']
    prob_pos = softmax_binary(z, model['c_neg'], model['c_pos'], T=model['T'])
    return z, prob_pos

def get_covid_split_values(df, split_feature='log_val'):
    if split_feature == 'log_val':
        return df['log_val'].values.astype(float)
    if split_feature == 'response':
        return df['response'].values.astype(float)
    if split_feature == 'sqrt_response':
        return np.sqrt(df['response'].values.astype(float))
    if split_feature == 'CLIA':
        return df['CLIA'].values.astype(float)
    raise ValueError(f'Unsupported split_feature: {split_feature}')

def fit_covid_hierarchical_model(train_df, bic_threshold=0.0, min_pos_samples=6,
                                 min_component_samples=2,
                                 split_feature=COVID_TERNARY_SPLIT_FEATURE):
    """Fit the COVID hierarchical model on the current training pool.

    Layer 1 is always binary (negative vs positive). Layer 2 becomes active only
    when the positive pool shows BIC support for a two-component split.
    """
    y_binary = train_df['label_binary'].values.astype(int)
    model = fit_binary_proto_model(train_df['log_val'].values, y_binary)

    split_values = get_covid_split_values(train_df, split_feature=split_feature)
    pos_split = split_values[y_binary == 1]

    split_active = False
    delta_bic = np.nan
    default_center = float(np.mean(pos_split)) if len(pos_split) else float(np.mean(split_values))
    c_weak = default_center
    c_strong = default_center
    T2 = 1.0

    if len(pos_split) >= min_pos_samples and np.nanstd(pos_split) > 1e-6:
        X_pos = pos_split.reshape(-1, 1)
        try:
            gmm1 = GaussianMixture(n_components=1, random_state=DEFAULT_SEED).fit(X_pos)
            gmm2 = GaussianMixture(n_components=2, random_state=DEFAULT_SEED).fit(X_pos)
            bic1 = float(gmm1.bic(X_pos))
            bic2 = float(gmm2.bic(X_pos))
            delta_bic = bic1 - bic2

            comps = gmm2.predict(X_pos)
            comp_counts = np.bincount(comps, minlength=2)
            means = gmm2.means_.flatten()
            order = np.argsort(means)

            if delta_bic > bic_threshold and np.min(comp_counts) >= min_component_samples:
                c_weak = float(means[order[0]])
                c_strong = float(means[order[1]])
                T2 = abs((c_strong - c_weak) / 3.0)
                if T2 < 1e-6:
                    T2 = 1.0
                split_active = True
        except ValueError:
            pass

    model.update({
        'split_active': split_active,
        'c_weak': c_weak,
        'c_strong': c_strong,
        'T2': T2,
        'delta_bic': delta_bic,
        'split_feature': split_feature,
    })
    return model

def predict_covid_hierarchical(model, test_df):
    log_values = test_df['log_val'].values.astype(float)
    split_values = get_covid_split_values(test_df, split_feature=model.get('split_feature', 'log_val'))
    z, prob_pos = predict_binary_proto(model, log_values)
    prob_neg = 1.0 - prob_pos
    prob_weak = np.zeros_like(prob_pos)
    prob_strong = prob_pos.copy()
    pred_label = np.zeros(len(z), dtype=int)

    pos_mask = prob_pos >= 0.5
    if model['split_active']:
        pw_cond, ps_cond = softmax_cond(split_values, model['c_weak'], model['c_strong'], T=model['T2'])
        prob_weak = prob_pos * pw_cond
        prob_strong = prob_pos * ps_cond
        pred_label[pos_mask] = np.where(prob_strong[pos_mask] >= prob_weak[pos_mask], 2, 1)
    else:
        pred_label[pos_mask] = 2

    return {
        'z': z,
        'prob_neg': prob_neg,
        'prob_pos': prob_pos,
        'prob_weak': prob_weak,
        'prob_strong': prob_strong,
        'pred_label': pred_label,
    }

def full_binary_model(df, label_col, feature_col='log_val'):
    model = fit_binary_proto_model(df[feature_col].values, df[label_col].values)
    cutoff = 0.5 * (model['c_neg'] + model['c_pos'])
    return model, cutoff

def merge_class_balanced_subject_order(pos_order, neg_order):
    pos_pool = list(pos_order)
    neg_pool = list(neg_order)
    ordered = []
    used_pos = used_neg = 0
    total_pos = len(pos_pool)
    total_neg = len(neg_pool)

    while pos_pool or neg_pool:
        next_len = len(ordered) + 1
        target_pos = next_len * total_pos / (total_pos + total_neg)
        target_neg = next_len * total_neg / (total_pos + total_neg)
        pos_gap = target_pos - used_pos if pos_pool else -1e9
        neg_gap = target_neg - used_neg if neg_pool else -1e9
        if pos_gap >= neg_gap:
            ordered.append(pos_pool.pop(0))
            used_pos += 1
        else:
            ordered.append(neg_pool.pop(0))
            used_neg += 1

    return ordered

def sample_subject_order(subjects, label_by_subject, rng, order_strategy='random'):
    if order_strategy == 'class_balanced':
        pos_subjects = [sid for sid in subjects if int(label_by_subject.loc[sid]) == 1]
        neg_subjects = [sid for sid in subjects if int(label_by_subject.loc[sid]) == 0]
        rng.shuffle(pos_subjects)
        rng.shuffle(neg_subjects)
        return np.array(merge_class_balanced_subject_order(pos_subjects, neg_subjects))
    if order_strategy == 'random':
        return rng.permutation(subjects)
    raise ValueError(f'Unsupported order_strategy: {order_strategy}')

def subject_cumulative_prequential_curve(df, label_col, group_col='subject_id', n_shuffles=DEFAULT_N_SHUFFLES,
                                         seed=DEFAULT_SEED, min_train_subjects=3,
                                         order_strategy='random', feature_col='log_val',
                                         require_both_classes=True):
    """Cumulative held-out prequential evaluation."""
    full_model, cutoff_final = full_binary_model(df, label_col, feature_col=feature_col)
    subjects = np.array(sorted(df[group_col].unique()))
    label_by_subject = df.groupby(group_col)[label_col].first().astype(int)
    x_grid = np.arange(min_train_subjects + 1, len(subjects) + 1)

    bacc_mat = np.full((n_shuffles, len(x_grid)), np.nan)
    cutdev_mat = np.full((n_shuffles, len(x_grid)), np.nan)
    rng = np.random.RandomState(seed)

    for s in range(n_shuffles):
        ordered = sample_subject_order(subjects, label_by_subject, rng, order_strategy=order_strategy)
        train_subjects = list(ordered[:min_train_subjects])
        remaining = ordered[min_train_subjects:]
        all_true, all_pred = [], []

        for sid in remaining:
            train_df = df[df[group_col].isin(train_subjects)]
            test_df = df[df[group_col] == sid]
            y_train = train_df[label_col].values.astype(int)
            if len(np.unique(y_train)) < 2:
                continue

            model = fit_binary_proto_model(train_df[feature_col].values, y_train)
            cutoff_now = 0.5 * (model['c_neg'] + model['c_pos'])
            _, p_test = predict_binary_proto(model, test_df[feature_col].values)
            y_test = test_df[label_col].values.astype(int)
            y_pred = (p_test >= 0.5).astype(int)

            all_true.extend(y_test.tolist())
            all_pred.extend(y_pred.tolist())
            total_seen = len(train_subjects) + 1
            idx = np.where(x_grid == total_seen)[0]
            if len(idx):
                idx = idx[0]
                if len(np.unique(all_true)) == 2:
                    bacc_mat[s, idx] = safe_balanced_accuracy(all_true, all_pred)
                elif not require_both_classes:
                    bacc_mat[s, idx] = np.mean(np.asarray(all_true) == np.asarray(all_pred))
                cutdev_mat[s, idx] = abs(cutoff_now - cutoff_final)
            train_subjects.append(sid)

    return x_grid, bacc_mat, cutdev_mat

def subject_stability_curve(df, label_col, group_col='subject_id', n_shuffles=DEFAULT_N_SHUFFLES,
                            seed=DEFAULT_SEED, min_subjects=3, order_strategy='random'):
    full_model, cutoff_final = full_binary_model(df, label_col)
    cpos_final = full_model['c_pos']
    subjects = np.array(sorted(df[group_col].unique()))
    label_by_subject = df.groupby(group_col)[label_col].first().astype(int)
    x_grid = np.arange(min_subjects, len(subjects) + 1)
    cpos_err_mat = np.full((n_shuffles, len(x_grid)), np.nan)
    cut_err_mat = np.full((n_shuffles, len(x_grid)), np.nan)
    rng = np.random.RandomState(seed)

    for s in range(n_shuffles):
        if order_strategy == 'class_balanced':
            pos_subjects = [sid for sid in subjects if int(label_by_subject.loc[sid]) == 1]
            neg_subjects = [sid for sid in subjects if int(label_by_subject.loc[sid]) == 0]
            rng.shuffle(pos_subjects)
            rng.shuffle(neg_subjects)
            perm = np.array(merge_class_balanced_subject_order(pos_subjects, neg_subjects))
        else:
            perm = rng.permutation(subjects)
        for i, n_subj in enumerate(x_grid):
            sub_df = df[df[group_col].isin(perm[:n_subj])]
            y = sub_df[label_col].values.astype(int)
            if len(np.unique(y)) < 2:
                continue
            model = fit_binary_proto_model(sub_df['log_val'].values, y)
            cutoff_now = 0.5 * (model['c_neg'] + model['c_pos'])
            cpos_err_mat[s, i] = abs(model['c_pos'] - cpos_final)
            cut_err_mat[s, i] = abs(cutoff_now - cutoff_final)

    return x_grid, cpos_err_mat, cut_err_mat

def subject_cumulative_hierarchical_curve(df, group_col='subject_id', n_shuffles=DEFAULT_N_SHUFFLES,
                                          seed=DEFAULT_SEED, min_train_subjects=10,
                                          order_strategy='class_balanced',
                                          split_feature=COVID_TERNARY_SPLIT_FEATURE,
                                          fallback_to_accuracy=True):
    """Track COVID evolution from binary-only to a BIC-triggered three-class model."""
    subjects = np.array(sorted(df[group_col].unique()))
    label_binary_by_subject = df.groupby(group_col)['label_binary'].first().astype(int)
    x_grid = np.arange(min_train_subjects + 1, len(subjects) + 1)

    binary_bacc_mat = np.full((n_shuffles, len(x_grid)), np.nan)
    ternary_bacc_mat = np.full((n_shuffles, len(x_grid)), np.nan)
    split_active_mat = np.full((n_shuffles, len(x_grid)), np.nan)

    rng = np.random.RandomState(seed)
    for s in range(n_shuffles):
        ordered = sample_subject_order(subjects, label_binary_by_subject, rng, order_strategy=order_strategy)
        train_subjects = list(ordered[:min_train_subjects])
        remaining = ordered[min_train_subjects:]

        all_true_binary, all_pred_binary = [], []
        all_true_ternary, all_pred_ternary = [], []

        for sid in remaining:
            train_df = df[df[group_col].isin(train_subjects)]
            test_df = df[df[group_col] == sid]
            y_train_binary = train_df['label_binary'].values.astype(int)
            if len(np.unique(y_train_binary)) < 2:
                continue

            model = fit_covid_hierarchical_model(train_df, split_feature=split_feature)
            pred = predict_covid_hierarchical(model, test_df)
            y_test_binary = test_df['label_binary'].values.astype(int)
            y_test_ternary = test_df['label_ternary'].values.astype(int)

            all_true_binary.extend(y_test_binary.tolist())
            all_pred_binary.extend((pred['prob_pos'] >= 0.5).astype(int).tolist())
            all_true_ternary.extend(y_test_ternary.tolist())
            all_pred_ternary.extend(pred['pred_label'].tolist())

            total_seen = len(train_subjects) + 1
            idx = np.where(x_grid == total_seen)[0]
            if len(idx):
                idx = idx[0]
                if len(np.unique(all_true_binary)) == 2:
                    binary_bacc_mat[s, idx] = safe_balanced_accuracy(all_true_binary, all_pred_binary)
                elif fallback_to_accuracy:
                    binary_bacc_mat[s, idx] = np.mean(
                        np.asarray(all_true_binary) == np.asarray(all_pred_binary)
                    )

                if len(np.unique(all_true_ternary)) >= 2:
                    ternary_bacc_mat[s, idx] = safe_balanced_accuracy(all_true_ternary, all_pred_ternary)
                elif fallback_to_accuracy:
                    ternary_bacc_mat[s, idx] = np.mean(
                        np.asarray(all_true_ternary) == np.asarray(all_pred_ternary)
                    )

                split_active_mat[s, idx] = float(model['split_active'])

            train_subjects.append(sid)

    return x_grid, binary_bacc_mat, ternary_bacc_mat, split_active_mat

def centered_moving_average(y, window=1):
    y = np.asarray(y, dtype=float)
    if window <= 1:
        return y.copy()

    valid = np.isfinite(y)
    if valid.sum() < 2:
        return y.copy()

    vals = y[valid]
    pad = window // 2
    vals_pad = np.pad(vals, (pad, pad), mode='edge')
    kernel = np.ones(window, dtype=float) / float(window)
    smoothed = np.convolve(vals_pad, kernel, mode='valid')
    if len(smoothed) != len(vals):
        smoothed = smoothed[:len(vals)]

    out = y.copy()
    out[valid] = smoothed
    return out

def bootstrap_median_ci(curves, n_boot=800, alpha=0.05, seed=DEFAULT_SEED):
    curves = np.asarray(curves, dtype=float)
    n_rows, n_cols = curves.shape
    rng = np.random.RandomState(seed)
    boot = np.full((n_boot, n_cols), np.nan)
    for b in range(n_boot):
        idx = rng.randint(0, n_rows, size=n_rows)
        boot[b] = np.nanmedian(curves[idx], axis=0)
    lo = np.nanpercentile(boot, 100 * (alpha / 2.0), axis=0)
    hi = np.nanpercentile(boot, 100 * (1.0 - alpha / 2.0), axis=0)
    return lo, hi

def bootstrap_mean_ci(curves, n_boot=800, alpha=0.05, seed=DEFAULT_SEED):
    curves = np.asarray(curves, dtype=float)
    n_rows, n_cols = curves.shape
    rng = np.random.RandomState(seed)
    boot = np.full((n_boot, n_cols), np.nan)
    for b in range(n_boot):
        idx = rng.randint(0, n_rows, size=n_rows)
        boot[b] = np.nanmean(curves[idx], axis=0)
    lo = np.nanpercentile(boot, 100 * (alpha / 2.0), axis=0)
    hi = np.nanpercentile(boot, 100 * (1.0 - alpha / 2.0), axis=0)
    return lo, hi

def plot_curve_only_panel(ax, x, curve, title, color, y_limits,
                          ci_label, ci_color, warm_start_subjects,
                          y_label='Balanced accuracy'):
    warm_fill_color = '#EFE9E2'
    warm_fill_alpha = 0.95
    ci_alpha = 0.35
    y_floor = 0.10
    y_pad = 0.02
    curve = np.asarray(curve, dtype=float)
    fancy_line(ax, x, curve, color=color, lw=2.5)
    ax.scatter(x, curve, s=30, facecolor='white', edgecolor=color, lw=1.2, zorder=4)

    if warm_start_subjects is not None:
        ax.axvspan(warm_start_subjects, warm_start_subjects + 1,
                   color=warm_fill_color or COLORS['gray'], alpha=warm_fill_alpha, zorder=0)
        ax.set_xlim(warm_start_subjects - 0.1, x.max() + 0.2)
    else:
        ax.set_xlim(x[0] - 0.5, x.max() + 0.5)

    valid = curve[np.isfinite(curve)]
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    elif valid.size:
        y0 = max(y_floor, float(np.nanmin(valid) - y_pad))
        y1 = min(1.02, float(np.nanmax(valid) + y_pad))
        if y1 - y0 < 0.06:
            mid = 0.5 * (y0 + y1)
            y0 = max(y_floor, mid - 0.03)
            y1 = min(1.02, mid + 0.03)
        ax.set_ylim(y0, y1)
    else:
        ax.set_ylim(y_floor, 1.02)

    if warm_start_subjects is not None:
        y0, y1 = ax.get_ylim()
        y_text = y1 - 0.06 * (y1 - y0)
        ax.text(warm_start_subjects + 0.50, y_text, 'warm start',
                ha='center', va='top', rotation=90,
                fontsize=6.5, color=COLORS['gray'])

    ax.set_xlabel('Cumulative labeled subjects', fontweight='bold')
    ax.set_ylabel(y_label, fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=11)
    
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color=color, lw=2.5, label=y_label),
        Patch(facecolor=ci_color, alpha=ci_alpha, edgecolor='none', label=ci_label)
    ]
    ax.legend(handles=legend_elements, frameon=True, fontsize=7.5, loc='lower right',
              facecolor='white', edgecolor='#DDDDDD', framealpha=0.95)

def plot_stability_panel(ax, x, mat, title, line_color, fill_color, label_text):
    ci_label = '95% CI (800 bootstraps)'
    ci_alpha = 0.30
    m = np.nanmedian(mat, axis=0)
    
    lo, hi = bootstrap_median_ci(mat, n_boot=800, alpha=0.05, seed=DEFAULT_SEED)
    ax.fill_between(x, lo, hi, color=fill_color, alpha=ci_alpha, zorder=1)

    fancy_line(ax, x, m, color=line_color, lw=2.2, zorder=3)
    ax.scatter(x, m, s=22, facecolor='white', edgecolor=line_color, lw=1.0, zorder=4)
    
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], color=line_color, lw=2.2, label=label_text),
        Patch(facecolor=fill_color, alpha=ci_alpha, edgecolor='none', label=ci_label)
    ]
    ax.legend(handles=legend_elements, frameon=True, fontsize=7.5, loc='upper right',
              bbox_to_anchor=(0.98, 0.98), facecolor='white', edgecolor='#DDDDDD', framealpha=0.95)
              
    ax.set_xlim(x.min() - 0.2, x.max() + 0.2)
    ax.set_ylim(bottom=0)
    ax.set_xlabel('Cumulative labeled subjects', fontweight='bold')
    ax.set_ylabel('Absolute deviation in z-space', fontweight='bold')
    ax.set_title(title, fontweight='bold', fontsize=10.5)
def plot_covid_sample_size_model_evolution(output='Suppl_COVID_SampleSize_ModelEvolution.png'):
    from matplotlib.patches import Patch
    import math
    sample_sizes = (10, 20, 30, 40)
    order_strategy = 'class_balanced'
    split_feature = COVID_TERNARY_SPLIT_FEATURE
    min_component_samples = 3
    top_row_label_y = -0.14
    bottom_row_label_y = -0.22

    df = load_covid_data(
        ternary_split_feature=split_feature,
        negative_model='uniform',
    )
    neg_reference = df.loc[df['label_binary'] == 0, 'log_val'].values.astype(float)
    df['z'], _, _ = zscore_normalize(df['log_val'].values.astype(float), neg_reference)

    subjects = np.array(sorted(df['subject_id'].unique()))
    label_binary_by_subject = df.groupby('subject_id')['label_binary'].first().astype(int)
    ordered = sample_subject_order(
        subjects,
        label_binary_by_subject,
        np.random.RandomState(DEFAULT_SEED),
        order_strategy=order_strategy,
    )

    z_min = float(df['z'].min() - 1.2)
    z_max = float(df['z'].max() + 1.2)
    z_grid = np.linspace(z_min, z_max, 500)

    n_panels = len(sample_sizes)
    ncols = 2
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(9.8, 7.0),
        sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes).ravel()

    panel_labels = 'abcdefghijklmnopqrstuvwxyz'
    class_markers = {
        0: ('o', COLORS['neg'], COLORS['neg_edge']),
        1: ('^', COLORS['weak'], COLORS['weak_edge']),
        2: ('D', COLORS['pos'], COLORS['pos_edge']),
    }

    for i, n in enumerate(sample_sizes):
        ax = axes[i]
        panel_df = df[df['subject_id'].isin(ordered[:n])].copy()
        panel_df = assign_covid_ternary_labels(
            panel_df,
            split_feature=split_feature,
            min_component_samples=min_component_samples,
        )

        neg_df = panel_df[panel_df['label_binary'] == 0]
        pos_df = panel_df[panel_df['label_binary'] == 1]
        split_active = bool((panel_df['label_ternary'] == 1).any() and (panel_df['label_ternary'] == 2).any())

        p_neg_proto = float(neg_df['z'].mean())
        p_pos_proto = float(pos_df['z'].mean())
        p_weak_proto = _proto_mean(panel_df, 1, p_pos_proto)
        p_strong_proto = _proto_mean(panel_df, 2, p_pos_proto)

        l1_scale = max(abs((p_pos_proto - p_neg_proto) / 3.0), 1e-6)
        l2_scale = max(abs((p_strong_proto - p_weak_proto) / 3.0), 1e-6)

        p_pos_curve = softmax_binary(z_grid, p_neg_proto, p_pos_proto, T=l1_scale)
        p_neg_curve = 1.0 - p_pos_curve
        if split_active:
            weak_cond, strong_cond = softmax_cond(z_grid, p_weak_proto, p_strong_proto, T=l2_scale)
            p_weak_curve = p_pos_curve * weak_cond
            p_strong_curve = p_pos_curve * strong_cond
            band_series = (p_neg_curve, p_weak_curve, p_strong_curve)
            band_colors = (COLORS['neg_fill'], COLORS['weak_fill'], COLORS['pos_fill'])
        else:
            p_weak_curve = np.zeros_like(p_pos_curve)
            p_strong_curve = p_pos_curve
            band_series = (p_neg_curve, p_strong_curve)
            band_colors = (COLORS['neg_fill'], COLORS['pos_fill'])

        ax.stackplot(z_grid, *band_series, colors=band_colors, alpha=0.95, zorder=1)
        ax.plot(z_grid, p_neg_curve, color='white', lw=0.9, zorder=2)
        if split_active:
            ax.plot(z_grid, p_neg_curve + p_weak_curve, color='white', lw=0.9, zorder=2)
        fancy_line(ax, z_grid, p_neg_curve, color=COLORS['neg_edge'], lw=1.2, zorder=3)
        if split_active:
            fancy_line(ax, z_grid, p_weak_curve, color=COLORS['weak_edge'], lw=1.2, zorder=3)
        fancy_line(ax, z_grid, p_strong_curve, color=COLORS['pos_edge'], lw=1.2, zorder=3)

        cut1_z = 0.5 * (p_neg_proto + p_pos_proto)
        ax.axvline(cut1_z, color=COLORS['gray'], lw=1.0, linestyle='--', zorder=4)
        fancy_text(
            ax,
            cut1_z + 0.18,
            0.96,
            f'Neg/pos cutoff\n{cut1_z:.1f}',
            color=COLORS['gray'],
            ha='left',
            va='top',
            fontsize=7.2,
            zorder=7,
        )

        if split_active:
            cut2_z = 0.5 * (p_weak_proto + p_strong_proto)
            ax.axvline(cut2_z, color=COLORS['gray'], lw=1.0, linestyle=':', zorder=4)
            fancy_text(
                ax,
                cut2_z + 0.18,
                0.96,
                f'Weak/strong cutoff\n{cut2_z:.1f}',
                color=COLORS['gray'],
                ha='left',
                va='top',
                fontsize=7.2,
                zorder=7,
            )

        labels_to_draw = [0, 1, 2] if split_active else [0, 2]
        for lbl in labels_to_draw:
            rows = panel_df[panel_df['label_ternary'] == lbl].copy()
            if rows.empty:
                continue
            vals = rows['z'].values.astype(float)
            offsets = _local_band_offsets(vals, collision_threshold=0.40)
            marker, fill, edge = class_markers[lbl]
            p_pos_val = softmax_binary(vals, p_neg_proto, p_pos_proto, T=l1_scale)
            p_neg_val = 1.0 - p_pos_val
            if split_active:
                weak_cond_val, strong_cond_val = softmax_cond(vals, p_weak_proto, p_strong_proto, T=l2_scale)
                p_weak_val = p_pos_val * weak_cond_val
                p_strong_val = p_pos_val * strong_cond_val
            else:
                p_weak_val = np.zeros_like(p_pos_val)
                p_strong_val = p_pos_val

            if lbl == 0:
                low = np.zeros_like(p_neg_val)
                high = p_neg_val
            elif lbl == 1:
                low = p_neg_val
                high = p_neg_val + p_weak_val
            else:
                low = p_neg_val + p_weak_val
                high = p_neg_val + p_weak_val + p_strong_val

            mid = 0.5 * (low + high)
            band_half = np.maximum(0.5 * (high - low), 1e-6)
            max_shift = np.minimum(0.025, 0.30 * band_half)
            step = np.minimum(0.008, 0.10 * band_half)
            y = mid + offsets * step
            y = np.minimum(np.maximum(y, low + max_shift * 0.35), high - max_shift * 0.35)

            ax.scatter(vals, y, s=22, facecolor=fill, edgecolor='white', marker=marker, lw=0.9, zorder=5)
            ax.scatter(vals, y, s=22, facecolor='none', edgecolor=edge, marker=marker, lw=0.8, zorder=6)

        ax.set_title(f'{n} subjects', fontweight='bold', fontsize=10.5, pad=5)
        ax.set_xlim(z_min, z_max)
        ax.set_ylim(0.0, 1.0)

        row_idx, col_idx = divmod(i, ncols)
        if row_idx == nrows - 1:
            ax.set_xlabel('Standardized signal z (NAb, sweat)', fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel('Class probability', fontweight='bold')

        label_y = top_row_label_y if row_idx == 0 else bottom_row_label_y
        add_panel_label(ax, panel_labels[i], y=label_y, fontsize=11.5)
    for j in range(n_panels, len(axes)):
        axes[j].set_visible(False)

    legend_handles = [
        Patch(facecolor=COLORS['neg_fill'], edgecolor=COLORS['neg_edge'], label='Negative'),
        Patch(facecolor=COLORS['weak_fill'], edgecolor=COLORS['weak_edge'], label='Weak positive'),
        Patch(facecolor=COLORS['pos_fill'], edgecolor=COLORS['pos_edge'], label='Positive or strong positive'),
    ]
    fig.legend(
        legend_handles,
        ['Negative', 'Weak positive', 'Positive or strong positive'],
        frameon=True,
        fontsize=7.6,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.958),
        ncol=3,
        facecolor='white',
        edgecolor='#DDDDDD',
        framealpha=0.95,
        columnspacing=1.5,
        handletextpad=0.6,
        borderpad=0.35,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.12,
        top=0.885,
        wspace=0.08,
        hspace=0.28,
    )

    fig.savefig(output, dpi=600)

def plot_suppl_learning_stability_combined(
        output='Suppl_Sy_HBV_COVID_LearningCurve_Stability.png',
        n_shuffles=DEFAULT_N_SHUFFLES):
    covid_order_strategy = 'class_balanced'
    covid_display_window = 3
    covid_min_train_subjects = 10
    hbv_min_train_subjects = 4
    hbv_y_limits = (0.70, 0.892)
    covid_y_limits = (0.37, 0.77)
    convergence_covid_min_subjects = 10

    covid_learning_df = load_covid_data(
        ternary_split_feature='CLIA',
        negative_model='truncated_noise',
    )

    hbv_x, hbv_bacc_mat, _ = subject_cumulative_prequential_curve(
        hbv_df, 'label', min_train_subjects=hbv_min_train_subjects, n_shuffles=n_shuffles,
        order_strategy='class_balanced', require_both_classes=False)
    hbv_curves = np.asarray([
        centered_moving_average(row, window=3)
        for row in np.asarray(hbv_bacc_mat, dtype=float)
    ], dtype=float)
    hbv_curve = np.nanmean(hbv_curves, axis=0)
    hbv_lo, hbv_hi = bootstrap_mean_ci(hbv_curves, n_boot=800, alpha=0.05, seed=DEFAULT_SEED)

    covid_x_single, _, covid_ternary_mat, _ = subject_cumulative_hierarchical_curve(
        covid_learning_df,
        min_train_subjects=covid_min_train_subjects,
        n_shuffles=n_shuffles,
        seed=DEFAULT_SEED,
        order_strategy=covid_order_strategy,
        split_feature='sqrt_response',
    )
    covid_curves = np.asarray([
        centered_moving_average(row, window=covid_display_window)
        for row in np.asarray(covid_ternary_mat, dtype=float)
    ], dtype=float)
    covid_curve = np.nanmedian(covid_curves, axis=0)
    covid_lo, covid_hi = bootstrap_median_ci(covid_curves, n_boot=800, alpha=0.05, seed=DEFAULT_SEED)
    show_mask = (covid_x_single > 11)

    hbv_x_conv, hbv_cpos_err_mat, _ = subject_stability_curve(
        hbv_df, 'label', min_subjects=3, n_shuffles=n_shuffles,
        order_strategy='class_balanced'
    )

    covid_x_conv, covid_cpos_err_mat, _ = subject_stability_curve(
        covid_df, 'label_binary', min_subjects=convergence_covid_min_subjects,
        n_shuffles=n_shuffles, order_strategy=covid_order_strategy
    )

    fig, axes = plt.subplots(2, 2, figsize=(10.8, 8.4))

    axes[0, 0].fill_between(
        hbv_x, hbv_lo, hbv_hi,
        color=COLORS['pos_fill'], alpha=0.35, zorder=1,
    )
    plot_curve_only_panel(
        axes[0, 0], hbv_x, hbv_curve,
        'HBV balanced accuracy',
        COLORS['pos_edge'],
        hbv_y_limits,
        '95% CI (800 bootstraps)',
        COLORS['pos_fill'],
        hbv_min_train_subjects,
        y_label='Balanced accuracy',
    )
    axes[0, 0].set_xlabel('Cumulative labeled subjects', fontweight='bold')


    axes[0, 1].fill_between(
        covid_x_single[show_mask], covid_lo[show_mask], covid_hi[show_mask],
        color=COLORS['weak'], alpha=0.35, zorder=1,
    )
    plot_curve_only_panel(
        axes[0, 1], covid_x_single[show_mask], covid_curve[show_mask],
        'COVID three-class balanced accuracy',
        COLORS['weak_edge'],
        covid_y_limits,
        '95% CI (800 bootstraps)',
        COLORS['weak'],
        covid_min_train_subjects + 1,
        y_label='Balanced accuracy',
    )
    axes[0, 1].set_xlim(covid_min_train_subjects + 1 - 0.1, covid_x_single.max() + 0.2)
    axes[0, 1].set_xlabel('Cumulative labeled subjects', fontweight='bold')

    plot_stability_panel(
        axes[1, 0],
        hbv_x_conv,
        hbv_cpos_err_mat,
        'HBV positive-prototype stabilization',
        COLORS['pos_edge'],
        COLORS['pos'],
        'Median |C$_{pos,n}$ - C$_{pos,final}$|'
    )

    plot_stability_panel(
        axes[1, 1],
        covid_x_conv,
        covid_cpos_err_mat,
        'COVID positive-prototype stabilization',
        COLORS['weak_edge'],
        COLORS['weak'],
        'Median |C$_{pos,n}$ - C$_{pos,final}$|'
    )

    add_panel_label(axes[0, 0], 'a', y=-0.14)
    add_panel_label(axes[0, 1], 'b', y=-0.14)
    add_panel_label(axes[1, 0], 'c', y=-0.14)
    add_panel_label(axes[1, 1], 'd', y=-0.14)

    fig.subplots_adjust(
        left=0.08,
        right=0.985,
        bottom=0.12,
        top=0.97,
        wspace=0.20,
        hspace=0.30,
    )
    fig.savefig(output, dpi=600)

if __name__ == '__main__':
    print('=' * 60)
    print('Generating 4 main figures ...')
    print('=' * 60)
    plot_fig_j_hbv()
    plot_fig_k_covid()
    plot_fig_m_bic()
    plot_fig_o_stacked()
    print('=' * 60)
    print('Generating supplement figures ...')
    print('=' * 60)
    plot_suppl_learning_stability_combined()
    plot_covid_sample_size_model_evolution()
    print('Done.')
    plt.show()
