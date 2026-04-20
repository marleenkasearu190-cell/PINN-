from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / 'data'
FIGURE_DIR = PROJECT_DIR / 'figures' / 'raw_download'
DAILY_FILE = DATA_DIR / 'processed' / 'ERA5_Acton_Daily_2025.csv'
HOURLY_FILE = DATA_DIR / 'processed' / 'ERA5_Acton_Hourly_2025.csv'
LST_FILE = DATA_DIR / 'raw' / 'Acton-LST-2025-MOD11A1-061-results.csv'
OUT_OVERVIEW = FIGURE_DIR / 'Acton_2025_overview.png'
OUT_TEMP_COMPARE = FIGURE_DIR / 'Acton_2025_temperature_comparison.png'
OUT_HOURLY_HEATMAP = FIGURE_DIR / 'Acton_2025_hourly_heatmap.png'
OUT_MONTHLY = FIGURE_DIR / 'Acton_2025_monthly_seasonal.png'
OUT_CORR = FIGURE_DIR / 'Acton_2025_correlation_matrix.png'


sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 200


MONTH_LABELS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
SEASON_MAP = {
    12: 'Winter', 1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring', 5: 'Spring',
    6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Autumn', 10: 'Autumn', 11: 'Autumn',
}
SEASON_ORDER = ['Winter', 'Spring', 'Summer', 'Autumn']


def load_data():
    daily = pd.read_csv(DAILY_FILE, parse_dates=['Date'])
    hourly = pd.read_csv(HOURLY_FILE, parse_dates=['Date'])
    lst = pd.read_csv(LST_FILE, parse_dates=['Date'])

    lst = lst[(lst['Date'] >= '2025-01-01') & (lst['Date'] <= '2025-12-31')].copy()
    lst['LST_K'] = pd.to_numeric(lst['MOD11A1_061_LST_Day_1km'], errors='coerce')
    lst['LST_C'] = lst['LST_K'] - 273.15
    lst['is_valid'] = lst['LST_K'] > 0

    qa_good = 'LST produced, good quality, not necessary to examine more detailed QA'
    qa_other = 'LST produced, other quality, recommend examination of more detailed QA'
    lst_valid = lst[lst['is_valid']].copy()
    lst_good = lst_valid[lst_valid['MOD11A1_061_QC_Day_MODLAND_Description'] == qa_good].copy()
    lst_other = lst_valid[lst_valid['MOD11A1_061_QC_Day_MODLAND_Description'] == qa_other].copy()

    merged = daily.merge(lst_valid[['Date', 'LST_C']], on='Date', how='left')

    daily['Month'] = daily['Date'].dt.month
    daily['MonthLabel'] = pd.Categorical([MONTH_LABELS[m - 1] for m in daily['Month']], categories=MONTH_LABELS, ordered=True)
    daily['Season'] = pd.Categorical(daily['Month'].map(SEASON_MAP), categories=SEASON_ORDER, ordered=True)

    valid = merged.dropna(subset=['LST_C']).copy()
    valid['Month'] = valid['Date'].dt.month
    valid['MonthLabel'] = pd.Categorical([MONTH_LABELS[m - 1] for m in valid['Month']], categories=MONTH_LABELS, ordered=True)
    valid['Season'] = pd.Categorical(valid['Month'].map(SEASON_MAP), categories=SEASON_ORDER, ordered=True)

    return daily, hourly, lst, lst_good, lst_other, merged, valid


def plot_overview(daily, lst_good, lst_other):
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)

    axes[0].plot(daily['Date'], daily['t2m_C'], label='ERA5 2m air temperature', color='#d95f02', linewidth=1.6)
    axes[0].plot(daily['Date'], daily['lblt_C'], label='ERA5 lake bottom temperature', color='#1b9e77', linewidth=1.6)
    axes[0].scatter(lst_good['Date'], lst_good['LST_C'], label='MODIS LST good QA', color='#1f78b4', s=18, alpha=0.85)
    axes[0].scatter(lst_other['Date'], lst_other['LST_C'], label='MODIS LST other QA', color='#7570b3', s=16, alpha=0.55)
    axes[0].set_ylabel('Temperature (C)')
    axes[0].set_title('Acton Lake 2025: ERA5 and MODIS overview')
    axes[0].legend(loc='upper right', ncol=2, fontsize=8)

    axes[1].plot(daily['Date'], daily['wind_norm_m_per_s'], color='#386cb0', linewidth=1.4)
    axes[1].fill_between(daily['Date'], daily['wind_norm_m_per_s'], color='#386cb0', alpha=0.16)
    axes[1].set_ylabel('Wind speed (m/s)')
    axes[1].set_title('10 m wind speed magnitude')

    axes[2].plot(daily['Date'], daily['Is_J_per_m2'], color='#e6ab02', linewidth=1.4)
    axes[2].fill_between(daily['Date'], daily['Is_J_per_m2'], color='#e6ab02', alpha=0.2)
    axes[2].set_ylabel('J/m2')
    axes[2].set_title('Surface solar radiation downwards')

    axes[3].plot(daily['Date'], daily['lmld_m'], color='#66a61e', linewidth=1.4)
    axes[3].fill_between(daily['Date'], daily['lmld_m'], color='#66a61e', alpha=0.2)
    axes[3].set_ylabel('Depth (m)')
    axes[3].set_title('Lake mixed layer depth')
    axes[3].set_xlabel('Date')

    fig.tight_layout()
    fig.savefig(OUT_OVERVIEW, bbox_inches='tight')
    plt.close(fig)


def plot_temperature_comparison(merged):
    valid = merged.dropna(subset=['LST_C']).copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    axes[0].plot(merged['Date'], merged['t2m_C'], label='ERA5 2m air temperature', color='#d95f02', linewidth=1.5)
    axes[0].plot(merged['Date'], merged['lblt_C'], label='ERA5 lake bottom temperature', color='#1b9e77', linewidth=1.5)
    axes[0].scatter(valid['Date'], valid['LST_C'], label='MODIS daytime LST', color='#1f78b4', s=20, alpha=0.9)
    axes[0].set_ylabel('Temperature (C)')
    axes[0].set_title('Temperature comparison on days with valid MODIS LST')
    axes[0].legend(loc='upper right')

    axes[1].scatter(valid['t2m_C'], valid['LST_C'], label='LST vs ERA5 air temperature', color='#d95f02', s=22, alpha=0.8)
    axes[1].scatter(valid['lblt_C'], valid['LST_C'], label='LST vs ERA5 bottom temperature', color='#1b9e77', s=22, alpha=0.7)
    if len(valid) > 1:
        corr_air = valid[['t2m_C', 'LST_C']].corr().iloc[0, 1]
        corr_bottom = valid[['lblt_C', 'LST_C']].corr().iloc[0, 1]
        axes[1].text(0.02, 0.98, f'corr(LST, t2m) = {corr_air:.2f}\ncorr(LST, lblt) = {corr_bottom:.2f}', transform=axes[1].transAxes, va='top', fontsize=10)
    axes[1].set_xlabel('ERA5 temperature (C)')
    axes[1].set_ylabel('MODIS LST (C)')
    axes[1].set_title('Pointwise comparison on valid LST days')
    axes[1].legend(loc='best')

    fig.tight_layout()
    fig.savefig(OUT_TEMP_COMPARE, bbox_inches='tight')
    plt.close(fig)


def plot_hourly_heatmap(hourly):
    hourly = hourly.copy()
    hourly['day_of_year'] = hourly['Date'].dt.dayofyear
    hourly['hour'] = hourly['Date'].dt.hour

    t2m_grid = hourly.pivot(index='hour', columns='day_of_year', values='t2m_C')
    wind_grid = hourly.pivot(index='hour', columns='day_of_year', values='wind_norm_m_per_s')

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    sns.heatmap(t2m_grid, ax=axes[0], cmap='coolwarm', cbar_kws={'label': 'C'})
    axes[0].set_title('Hourly 2 m air temperature heatmap')
    axes[0].set_ylabel('Hour of day')

    sns.heatmap(wind_grid, ax=axes[1], cmap='YlGnBu', cbar_kws={'label': 'm/s'})
    axes[1].set_title('Hourly 10 m wind speed heatmap')
    axes[1].set_ylabel('Hour of day')
    axes[1].set_xlabel('Day of year')

    fig.tight_layout()
    fig.savefig(OUT_HOURLY_HEATMAP, bbox_inches='tight')
    plt.close(fig)


def plot_monthly_seasonal(daily, valid):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    monthly = daily.groupby('MonthLabel', observed=False)[['t2m_C', 'lblt_C', 'wind_norm_m_per_s', 'Is_J_per_m2']].mean().reset_index()
    monthly_melt = monthly.melt(id_vars='MonthLabel', var_name='Variable', value_name='Value')
    name_map = {
        't2m_C': 'ERA5 2m air temp',
        'lblt_C': 'ERA5 bottom temp',
        'wind_norm_m_per_s': 'Wind speed',
        'Is_J_per_m2': 'Solar radiation',
    }
    monthly_melt['Variable'] = monthly_melt['Variable'].map(name_map)
    sns.lineplot(data=monthly_melt, x='MonthLabel', y='Value', hue='Variable', marker='o', ax=axes[0, 0])
    axes[0, 0].set_title('Monthly mean seasonal cycle')
    axes[0, 0].set_xlabel('Month')
    axes[0, 0].tick_params(axis='x', rotation=35)

    sns.boxplot(data=daily, x='MonthLabel', y='wind_norm_m_per_s', color='#80b1d3', ax=axes[0, 1])
    axes[0, 1].set_title('Monthly distribution of wind speed')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Wind speed (m/s)')
    axes[0, 1].tick_params(axis='x', rotation=35)

    season_temp = valid.melt(
        id_vars=['Season'],
        value_vars=['LST_C', 't2m_C', 'lblt_C'],
        var_name='Variable',
        value_name='Temperature_C'
    )
    season_temp['Variable'] = season_temp['Variable'].map({
        'LST_C': 'MODIS LST',
        't2m_C': 'ERA5 2m air temp',
        'lblt_C': 'ERA5 bottom temp',
    })
    sns.boxplot(data=season_temp, x='Season', y='Temperature_C', hue='Variable', ax=axes[1, 0])
    axes[1, 0].set_title('Seasonal temperature distribution on valid LST days')
    axes[1, 0].set_xlabel('Season')
    axes[1, 0].set_ylabel('Temperature (C)')

    lst_count = valid.groupby('MonthLabel', observed=False).size().reindex(MONTH_LABELS, fill_value=0)
    axes[1, 1].bar(lst_count.index, lst_count.values, color='#8da0cb')
    axes[1, 1].set_title('Monthly count of valid MODIS LST observations')
    axes[1, 1].set_xlabel('Month')
    axes[1, 1].set_ylabel('Valid observation days')
    axes[1, 1].tick_params(axis='x', rotation=35)

    fig.tight_layout()
    fig.savefig(OUT_MONTHLY, bbox_inches='tight')
    plt.close(fig)


def plot_correlation_matrix(valid):
    corr_df = valid[['LST_C', 't2m_C', 'lblt_C', 'wind_norm_m_per_s', 'Is_J_per_m2', 'lmld_m']].copy()
    corr_df = corr_df.rename(columns={
        'LST_C': 'MODIS LST',
        't2m_C': 'ERA5 2m air temp',
        'lblt_C': 'ERA5 bottom temp',
        'wind_norm_m_per_s': 'Wind speed',
        'Is_J_per_m2': 'Solar radiation',
        'lmld_m': 'Mixed layer depth',
    })

    grid = sns.pairplot(
        corr_df,
        corner=True,
        diag_kind='hist',
        plot_kws={'s': 24, 'alpha': 0.65, 'edgecolor': 'none'},
        diag_kws={'bins': 18, 'color': '#4c72b0'}
    )
    grid.fig.suptitle('Correlation matrix on valid MODIS LST days', y=1.02)
    grid.fig.savefig(OUT_CORR, bbox_inches='tight')
    plt.close(grid.fig)


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    daily, hourly, lst, lst_good, lst_other, merged, valid = load_data()
    plot_overview(daily, lst_good, lst_other)
    plot_temperature_comparison(merged)
    plot_hourly_heatmap(hourly)
    plot_monthly_seasonal(daily, valid)
    plot_correlation_matrix(valid)

    print('可视化完成:')
    print(OUT_OVERVIEW)
    print(OUT_TEMP_COMPARE)
    print(OUT_HOURLY_HEATMAP)
    print(OUT_MONTHLY)
    print(OUT_CORR)
    print(f'MODIS 2025 有效 LST 天数: {len(valid)} / 365')


if __name__ == '__main__':
    main()
