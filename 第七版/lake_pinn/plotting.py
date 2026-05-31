"""Plotting helpers for annual and monthly lake temperature diagnostics."""

import calendar

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _infer_year_label(dates, metadata=None):
    if metadata is not None and metadata.get('year') is not None:
        try:
            return str(int(metadata['year']))
        except (TypeError, ValueError):
            return str(metadata['year'])
    date_values = pd.to_datetime(dates, errors='coerce').dropna()
    if date_values.empty:
        return ''
    years = sorted(date_values.dt.year.unique().tolist())
    if len(years) == 1:
        return str(int(years[0]))
    return f'{int(years[0])}-{int(years[-1])}'


def _lake_year_title(lake_name, dates, metadata=None):
    lake_name = str(lake_name or 'Lake').strip()
    year_label = _infer_year_label(dates, metadata=metadata)
    if year_label and year_label not in lake_name:
        return f'{lake_name} ({year_label})'
    return lake_name


def plot_year_heatmap(df, temp_grid, depths, output_path, metadata):
    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    day_axis = df['full_doy'].to_numpy()
    vmin = float(np.nanmin(temp_grid))
    vmax = float(np.nanmax(temp_grid))
    filled_levels = np.linspace(vmin, vmax, 28)
    line_levels = np.arange(np.floor(vmin / 4.0) * 4.0, np.ceil(vmax / 4.0) * 4.0 + 0.1, 4.0)
    if line_levels.size < 2:
        line_levels = np.linspace(vmin, vmax, 6)

    image = ax.contourf(day_axis, depths, temp_grid, levels=filled_levels, cmap='RdYlBu_r', extend='both')
    contour_lines = ax.contour(day_axis, depths, temp_grid, levels=line_levels, colors='black', linewidths=1.1, alpha=0.45)
    ax.clabel(contour_lines, fmt='%d', fontsize=10, inline=True)

    month_midpoints = df.groupby(df['Date'].dt.month)['full_doy'].mean()
    ax.set_xticks(month_midpoints.values)
    ax.set_xticklabels([calendar.month_abbr[month] for month in month_midpoints.index], fontsize=17)
    ax.set_xlabel('Month', fontsize=20, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=18, fontweight='bold')
    lake_title = _lake_year_title(metadata.get('lake_name', 'Lake'), df['Date'], metadata=metadata)
    ax.set_title(f"Annual Water Temperature Profile of {lake_title}", fontsize=26)
    ax.set_ylim(depths[-1], depths[0])
    ax.tick_params(axis='y', labelsize=15)

    max_depth = float(depths[-1])
    ax.text(25, max_depth * 0.10, 'Winter\nInverse\nStratification', color='blue', fontsize=22, fontweight='bold', ha='center')
    ax.text(120, max_depth * 0.88, 'Spring\nWarming', color='green', fontsize=22, fontweight='bold', ha='center')
    ax.text(210, max_depth * 0.78, 'Summer\nStratification\n(Thermocline)', color='red', fontsize=24, fontweight='bold', ha='center')
    ax.text(305, max_depth * 0.52, 'Autumn\nOverturn\n(Homothermal)', color='black', fontsize=22, fontweight='bold', ha='center')

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label('Temperature (C)', fontsize=20)
    cbar.ax.tick_params(labelsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_time_depth_curves(df, temp_grid, depths, output_path, metadata):
    fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)
    dates = pd.to_datetime(df['Date'])
    day_numbers = (dates - dates.min()).dt.days.values
    temp_min = np.nanmin(temp_grid)
    temp_max = np.nanmax(temp_grid)
    norm = plt.Normalize(vmin=temp_min, vmax=temp_max)

    step = 5
    for i in range(0, len(day_numbers), step):
        temp_profile = temp_grid[:, i]
        color = plt.cm.RdYlBu_r(norm(np.mean(temp_profile)))
        linewidth = 1.5 if (i % 30 == 0) else 0.8
        ax.plot(temp_profile, depths, color=color, linewidth=linewidth, alpha=0.7)

    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label('Temperature (C)', fontsize=16, fontweight='bold')
    cbar.ax.tick_params(labelsize=14)

    ax.set_xlabel('Temperature (C)', fontsize=18, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=18, fontweight='bold')
    ax.set_title(f"Time-Depth Temperature Profiles: {metadata['lake_name']} {metadata['year']}", fontsize=22, fontweight='bold')
    ax.set_ylim(depths[-1], depths[0])
    ax.grid(True, alpha=0.3, linestyle=':')

    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def plot_monthly_heatmaps(df, temp_grid, depths, output_path, metadata):
    dates = pd.to_datetime(df['Date'])
    temp_df = pd.DataFrame(
        {
            'Date': dates,
            'Month': dates.dt.month,
            'Day': dates.dt.day,
        }
    )

    monthly_day_max = temp_df.groupby('Month')['Day'].max().reindex(range(1, 13), fill_value=31)
    vmin = float(np.nanmin(temp_grid))
    vmax = float(np.nanmax(temp_grid))

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), constrained_layout=True)
    pcm = None

    for month_idx, ax in enumerate(axes.flat, start=1):
        month_mask = temp_df['Month'] == month_idx
        month_rows = temp_df.loc[month_mask]
        if month_rows.empty:
            ax.set_title(calendar.month_abbr[month_idx], fontsize=14, fontweight='bold')
            ax.set_xlim(1, monthly_day_max.loc[month_idx])
            ax.set_ylim(depths[-1], depths[0])
            ax.grid(True, alpha=0.35)
            continue

        month_grid = temp_grid[:, month_mask.to_numpy()]
        day_values = month_rows['Day'].to_numpy(dtype=np.float32)
        x_edges = np.arange(0.5, monthly_day_max.loc[month_idx] + 1.5, 1.0, dtype=np.float32)
        if len(depths) > 1:
            depth_step = np.diff(depths)
            top_edge = max(0.0, depths[0] - depth_step[0] / 2.0)
            bottom_edge = depths[-1] + depth_step[-1] / 2.0
            inner_edges = 0.5 * (depths[:-1] + depths[1:])
            y_edges = np.concatenate([[top_edge], inner_edges, [bottom_edge]])
        else:
            y_edges = np.array([0.0, max(float(depths[0]), 1.0)], dtype=np.float32)

        padded_grid = np.full((len(depths), len(x_edges) - 1), np.nan, dtype=np.float32)
        for col_idx, day_value in enumerate(day_values.astype(int)):
            padded_grid[:, day_value - 1] = month_grid[:, col_idx]

        pcm = ax.pcolormesh(
            x_edges,
            y_edges,
            padded_grid,
            cmap='turbo',
            vmin=vmin,
            vmax=vmax,
            shading='flat',
        )
        ax.set_title(calendar.month_abbr[month_idx], fontsize=14, fontweight='bold')
        ax.set_xlim(1, monthly_day_max.loc[month_idx])
        ax.set_ylim(depths[-1], depths[0])
        ax.set_xticks(np.arange(5, monthly_day_max.loc[month_idx] + 1, 5))
        ax.grid(True, alpha=0.35, color='white', linewidth=0.8)
        if month_idx in (1, 5, 9):
            ax.set_ylabel('Depth (m)')
        else:
            ax.set_yticklabels([])
        if month_idx >= 9:
            ax.set_xlabel('Day of month')

    fig.suptitle(
        f"{metadata['lake_name']} {metadata['year']} Monthly Temperature-Depth Heatmaps",
        fontsize=18,
    )
    cbar = fig.colorbar(pcm, ax=axes, shrink=0.94, pad=0.02)
    cbar.set_label('Temperature (C)')
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
