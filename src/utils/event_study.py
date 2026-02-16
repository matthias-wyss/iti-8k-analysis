import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl

import polars as pl
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def event_study(df_events,
                window_before=10,
                window_after=10,
                est_window=500,
                gap=30,
                min_obs=250,
                market_model='fama-mom',
                subject = 'returns'):
    """
    Generic event study function (Fama-French + Momentum).
    
    Parameters
    ----------
    df_events : pd.DataFrame
        Must contain columns ['permno', 'event_date'] where event_date
        is already a datetime or convertible to datetime.
    window_before, window_after : int
        Number of calendar days before/after the event to include.
    est_window : int
        Length (in calendar days) of the estimation window.
    gap : int
        Gap (in calendar days) between the end of estimation window and event date.
    min_obs : int
        Minimum number of usable observations in the estimation window.
    market_model : str
        Currently only 'fama-mom' is implemented.
    subject : str
        'returns', 'ITI' or 'abs returns' to specify which data to use for the event study.
    """

    if market_model != 'fama-mom':
        raise ValueError("Only 'fama-mom' is implemented for now.")

    df_events = df_events.copy()
    df_events['event_date'] = pd.to_datetime(df_events['event_date'])

    valid_permnos = df_events['permno'].unique()

    #Load CRSP returns (polars for speed, then to pandas)
    if subject == 'returns':
        returns = pl.read_csv('./data/raw/crsp_daily_us.csv')
        returns = returns.filter(pl.col('permno').is_in(valid_permnos))
        df_returns = returns.to_pandas()[['permno', 'date', 'ret']]
        df_returns['date'] = pd.to_datetime(df_returns['date'])
    elif subject == 'abs returns' :
        returns = pl.read_csv('./data/raw/crsp_daily_us.csv')
        returns = returns.filter(pl.col('permno').is_in(valid_permnos))
        df_returns = returns.to_pandas()[['permno', 'date', 'ret']]
        df_returns['date'] = pd.to_datetime(df_returns['date'])
        df_returns['ret'] = df_returns['ret'].abs()
    elif subject == 'ITI' : 
        returns = pl.read_csv('./data/raw/ITIs.csv')
        returns = returns.filter(pl.col('permno').is_in(valid_permnos))
        df_returns = returns.to_pandas()[['permno', 'date', 'ITI(13D)']]
        df_returns = df_returns.rename(columns={'ITI(13D)':'ret'})
        df_returns['date'] = pd.to_datetime(df_returns['date'])
    else : 
        raise ValueError("Subject must be either 'returns', 'abs returns' or 'ITI'.")
    
    # Load Fama-French + Momentum factors
    df_factors = pd.read_csv('./data/raw/ff_daily_factors.csv')
    df_factors['date'] = pd.to_datetime(df_factors['date'])
    df_factors['MKT_RF'] = df_factors['MKT'] - df_factors['RF']

    # Group returns by permno once
    returns_by_permno = {p: g.sort_values('date') for p, g in df_returns.groupby('permno')}

    out = []

    for _, row in df_events.iterrows():
        permno = row['permno']
        event_date = row['event_date']

        if permno not in returns_by_permno:
            continue

        df = returns_by_permno[permno]

        # ---------- Estimation window ----------
        est_start = event_date - pd.Timedelta(days=est_window + gap + 1)
        est_end   = event_date - pd.Timedelta(days=gap + 1)

        df_est = df[(df['date'] >= est_start) & (df['date'] <= est_end)].copy()
        df_est = df_est.dropna(subset=['ret'])

        if len(df_est) < min_obs:
            continue

        # Merge with factors
        df_merged_est = df_est.merge(df_factors, on='date', how='inner')
        df_merged_est = df_merged_est.dropna(subset=['ret', 'MKT_RF', 'SMB', 'HML', 'MOM'])

        if len(df_merged_est) < min_obs:
            continue

        X = df_merged_est[['MKT_RF', 'SMB', 'HML', 'MOM']]
        y = df_merged_est['ret']
        reg = LinearRegression().fit(X, y)

        # ---------- Event window ----------
        ev_start = event_date - pd.Timedelta(days=window_before)
        ev_end   = event_date + pd.Timedelta(days=window_after)

        df_ev = df[(df['date'] >= ev_start) & (df['date'] <= ev_end)].copy()
        df_ev = df_ev.dropna(subset=['ret'])
        if df_ev.empty:
            continue

        df_ev = df_ev.merge(df_factors, on='date', how='left')
        df_ev = df_ev.dropna(subset=['MKT_RF','SMB','HML','MOM'])
        if df_ev.empty:
            continue
        df_ev['exp_ret'] = reg.predict(df_ev[['MKT_RF','SMB','HML','MOM']])
        df_ev['AR'] = df_ev['ret'] - df_ev['exp_ret']
        df_ev['tau'] = (df_ev['date'] - event_date).dt.days

        df_ev['permno'] = permno
        df_ev['event_date'] = event_date

        out.append(df_ev[['permno','event_date','date','tau','ret','exp_ret','AR']])

    if not out:
        raise ValueError("No valid events after filtering. Check your windows and data.")

    results = pd.concat(out, ignore_index=True)

    if subject == 'ITI' :
        results = results.sort_values(['permno','event_date','tau'])
        results['CAR'] = results['AR']
    else : 
        # compute CAR per event
        results = results.sort_values(['permno','event_date','tau'])
        results['CAR'] = results.groupby(['permno','event_date'])['AR'].cumsum()

    return results


def compute_caar_ci(df, tau_col='tau'):
    """
    df must contain columns: tau_col, CAR
    returns DataFrame with columns:
    tau, mean_CAR, ci_low, ci_high
    """
    z = 1.96

    agg = (
        df.groupby(tau_col)['CAR']
          .agg(['mean', 'std', 'count'])
          .rename(columns={'mean': 'mean_CAR',
                           'std': 'std_CAR',
                           'count': 'n'})
          .reset_index()
    )

    agg['se_CAR'] = agg['std_CAR'] / np.sqrt(agg['n'])
    agg['ci_low'] = agg['mean_CAR'] - z * agg['se_CAR']
    agg['ci_high'] = agg['mean_CAR'] + z * agg['se_CAR']

    return agg

def plot_caar_ci(
    agg,
    tau_col='tau',
    title='Cumulative Average Abnormal Returns (CAAR)',
    subject='returns',
    ax=None,
    ylim=None,
    save_as: str = None
):

    # Determine label
    label = 'CAAR' if subject == 'returns' else 'Average AI'

    # If no axis provided, create a standalone figure (backward compatible)
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
        created_fig = True

    # Plot CI region
    ax.fill_between(
        agg[tau_col], agg['ci_low'], agg['ci_high'],
        alpha=0.2, label='95% CI'
    )

    # Plot mean CAAR / mean CAR
    ax.plot(
        agg[tau_col], agg['mean_CAR'],
        linewidth=2, label=label
    )

    # Event lines
    ax.axvline(0, linestyle='--')
    ax.axhline(0, linestyle=':', lw=1)

    # Labels
    ax.set_xlabel("Event time τ (days)")
    ax.set_ylabel("Average AI" if subject == 'ITI' else "CAAR")
    ax.set_title(title)

    # Y-limits
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    # Cosmetics
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Only show when the function created the figure
    if created_fig:
        plt.tight_layout()
        if save_as is not None:
            plt.savefig(save_as, dpi=300)
        plt.show()

def plot_x_vs_y(results_report, results_filing, x_title = '', y_title = ''):
    agg_rep = compute_caar_ci(results_report, tau_col='tau')
    agg_fil = compute_caar_ci(results_filing, tau_col='tau')

    fig, axes = plt.subplots(1, 2, figsize=(14,5), sharey=True)

    # Panel A: report
    ax = axes[0]
    ax.fill_between(agg_rep['tau'], agg_rep['ci_low'], agg_rep['ci_high'], alpha=0.2)
    ax.plot(agg_rep['tau'], agg_rep['mean_CAR'], lw=2)
    ax.axvline(0, linestyle='--')
    ax.axhline(0, linestyle=':', lw=1)
    ax.set_title(x_title)
    ax.set_xlabel('τ (days)')
    ax.set_ylabel('CAAR')
    ax.grid(True, alpha=0.3)

    # Panel B: filing
    ax = axes[1]
    ax.fill_between(agg_fil['tau'], agg_fil['ci_low'], agg_fil['ci_high'], alpha=0.2)
    ax.plot(agg_fil['tau'], agg_fil['mean_CAR'], lw=2)
    ax.axvline(0, linestyle='--')
    ax.axhline(0, linestyle=':', lw=1)
    ax.set_title(y_title)
    ax.set_xlabel('τ (days)')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()



import numpy as np
import polars as pl
import pandas as pd

np.random.seed(42)

def make_random_events(df_events,
                       crsp_path='./data/raw/crsp_daily_us.csv',
                       buffer_days=60,
                       n_random_per_event=1):
    """
    Generate placebo events:
    - Same permnos as df_events
    - For each real event, draw `n_random_per_event` random trading dates
      for that permno.
    - Exclude a +/- buffer_days window around *any* real event of that permno
      so we don't accidentally pick true event periods.
    
    df_events: pandas DataFrame with ['permno','event_date'] (datetime)
    """

    df_events = df_events.copy()
    df_events['event_date'] = pd.to_datetime(df_events['event_date'])
    df_events['permno'] = df_events['permno'].astype(int)

    unique_permnos = df_events['permno'].unique()

    # Load CRSP dates for those permnos (only need permno + date)
    crsp = (
        pl.read_csv(crsp_path, columns=['permno','date'])
          .filter(pl.col('permno').is_in(unique_permnos))
    )
    df_crsp = crsp.to_pandas()
    df_crsp['date'] = pd.to_datetime(df_crsp['date'])
    df_crsp['permno'] = df_crsp['permno'].astype(int)

    # Precompute trading dates per permno
    dates_by_permno = {
        p: g['date'].sort_values().reset_index(drop=True)
        for p, g in df_crsp.groupby('permno')
    }

    # Keep only events for permnos that actually exist in CRSP
    available_permnos = set(dates_by_permno.keys())
    df_events = df_events[df_events['permno'].isin(available_permnos)]
    unique_permnos = df_events['permno'].unique()

    # For each permno, build a set of "forbidden" dates around real events
    forbidden = {}
    for p in unique_permnos:
        sub = dates_by_permno.get(p)
        if sub is None or sub.empty:
            # No trading dates for this permno -> skip
            continue

        ev_dates = df_events.loc[df_events['permno'] == p, 'event_date'].unique()
        forbidden_dates = set()
        for d in ev_dates:
            lo = d - pd.Timedelta(days=buffer_days)
            hi = d + pd.Timedelta(days=buffer_days)
            # all trading days within [lo, hi]
            mask = (sub >= lo) & (sub <= hi)
            forbidden_dates.update(sub[mask].tolist())
        forbidden[p] = forbidden_dates

    # Now sample random dates per event
    rows = []
    for _, row in df_events.iterrows():
        p = int(row['permno'])
        if p not in dates_by_permno:
            # Safety check, should already be filtered out above
            continue

        all_dates = dates_by_permno[p]

        # allowed trading dates = all minus forbidden window
        forb = forbidden.get(p, set())
        allowed = all_dates[~all_dates.isin(list(forb))]
        if allowed.empty:
            # No admissible placebo date for this event
            continue

        # sample n_random_per_event dates (with replacement is fine for placebo)
        sampled = np.random.choice(allowed.values, size=n_random_per_event, replace=True)

        for d_rand in sampled:
            rows.append({'permno': p, 'event_date': pd.to_datetime(d_rand)})

    df_random_events = pd.DataFrame(rows).drop_duplicates()
    return df_random_events


def plot_corr_abn_iti_ar(results_returns :pd.DataFrame, results_iti : pd.DataFrame, title: str, save_as: str = None):
    """
    Plot correlation between average AI and average AR at each event time τ.
    """

    results_iti.rename(columns={'AR': 'ITI_AR'}, inplace=True)
    corr = results_returns.merge(
        results_iti,
        on=['permno','event_date','tau'],
        how = 'inner')

    corr_tau = (
        corr.groupby("tau")[["AR", "ITI_AR"]]
                .corr()
                .iloc[0::2, -1]        
                .reset_index()
                .rename(columns={"ITI_AR": "corr"})
    )

    corr_tau = corr_tau[['tau', 'corr']]  

    corr_tau.rename(columns={"ITI_AR": "corr"}, inplace=True)

    corr_matrix = corr_tau.pivot_table(columns='tau', values='corr')
    corr_matrix.index = ["corr(average AI, average AR)"]

    plt.figure(figsize=(14, 2.5))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        vmin=-0.181,
        vmax=0.204,
        center=0,
        cbar_kws={'label': 'Correlation'}
    )

    plt.title(title)
    plt.xlabel("Event Time τ")
    plt.tight_layout()
    if save_as is not None:
        plt.savefig(save_as, dpi=300)
    plt.show()