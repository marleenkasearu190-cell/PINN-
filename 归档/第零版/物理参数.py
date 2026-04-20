import argparse
import calendar
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_ERA5_PATH = str(PROJECT_DIR / 'ERA5_mendota_2018_Daily.csv')
DEFAULT_LST_PATH = str(PROJECT_DIR / 'Lake-Mendota-MOD11A1-061-results.csv')


class LakePINN(nn.Module):
    def __init__(self, hidden_dim=64, hidden_layers=4):
        super().__init__()
        layers = [nn.Linear(2, hidden_dim), nn.Softplus()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Softplus()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.net(inputs)


def sanitize_name(text: str) -> str:
    return re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')


def infer_metadata(merged: pd.DataFrame, lst: pd.DataFrame, era5_path: Path, lst_path: Path):
    year = int(merged['Date'].dt.year.mode().iloc[0])

    lake_name = None
    for col in ['ID', 'Category']:
        if col in lst.columns:
            values = lst[col].dropna().astype(str)
            values = values[values.str.strip() != '']
            if not values.empty:
                lake_name = values.iloc[0].strip()
                break

    if not lake_name:
        lake_name = lst_path.stem.replace('-', ' ').replace('_', ' ')

    base_tag = sanitize_name(lake_name)
    if not base_tag:
        base_tag = sanitize_name(era5_path.stem.replace('-', ' ').replace('_', ' '))
    file_tag = f'{base_tag}_{year}'

    return {'lake_name': lake_name, 'year': year, 'file_tag': file_tag}


def load_training_frame(era5_path, lst_path):
    era5_path = Path(era5_path)
    lst_path = Path(lst_path)

    if not era5_path.exists():
        raise FileNotFoundError(
            f'ERA5 file not found: {era5_path}. Please check the --era5 path or place the file in the expected location.'
        )
    if not lst_path.exists():
        raise FileNotFoundError(
            f'LST file not found: {lst_path}. Please check the --lst path or place the file in the expected location.'
        )

    era5 = pd.read_csv(era5_path)
    era5['Date'] = pd.to_datetime(era5['Date'])
    era5 = era5.sort_values('Date').copy()

    lst = pd.read_csv(lst_path)
    lst['Date'] = pd.to_datetime(lst['Date'])
    lst = lst.sort_values('Date').copy()
    lst['LST_surface_K'] = pd.to_numeric(lst['MOD11A1_061_LST_Day_1km'], errors='coerce')
    lst.loc[lst['LST_surface_K'] <= 0, 'LST_surface_K'] = np.nan

    lst_daily = lst.groupby('Date', as_index=False)['LST_surface_K'].mean().sort_values('Date')

    merged = era5.merge(lst_daily, on='Date', how='left')
    merged = merged.sort_values('Date').copy()
    merged['day_index'] = np.arange(len(merged), dtype=np.float32)
    merged['full_doy'] = merged['Date'].dt.dayofyear.astype(np.float32)
    merged['month'] = merged['Date'].dt.month

    merged['LST_surface_K'] = (
        merged['LST_surface_K']
        .interpolate(method='linear', limit_direction='both')
        .bfill()
        .ffill()
    )
    merged['LST_surface_C'] = merged['LST_surface_K'] - 273.15
    merged['BottomTemp_C'] = pd.to_numeric(merged['lblt_C'], errors='coerce')
    merged['Solar_J_m2'] = pd.to_numeric(merged['Is_J_per_m2'], errors='coerce')
    merged['MixedLayerDepth_m'] = pd.to_numeric(merged['lmld_m'], errors='coerce')
    merged['T_air_C'] = pd.to_numeric(merged['t2m_C'], errors='coerce')
    merged['wind_norm'] = pd.to_numeric(merged['wind_norm_m_per_s'], errors='coerce')

    if merged['T_air_C'].isna().all():
        merged['T_air_C'] = merged['LST_surface_C']
    else:
        merged['T_air_C'] = merged['T_air_C'].interpolate(method='linear', limit_direction='both').bfill().ffill()

    if merged['wind_norm'].isna().all():
        merged['wind_norm'] = 1.0
    else:
        merged['wind_norm'] = merged['wind_norm'].interpolate(method='linear', limit_direction='both').bfill().ffill()

    required = ['LST_surface_C', 'BottomTemp_C', 'Solar_J_m2', 'MixedLayerDepth_m', 'T_air_C', 'wind_norm']
    if merged[required].isna().any().any():
        raise ValueError('Input data still contains missing values after preprocessing.')

    metadata = infer_metadata(merged, lst, era5_path, lst_path)
    return merged, metadata


def compute_losses(model, batch, max_depth):
    t_col = batch['t_colloc'].clone().detach().requires_grad_(True)
    z_col = batch['z_colloc'].clone().detach().requires_grad_(True)
    model_input = torch.cat([t_col, z_col / max_depth], dim=1)
    T_pred = model(model_input)

    dT_dt = torch.autograd.grad(T_pred, t_col, grad_outputs=torch.ones_like(T_pred), create_graph=True)[0]
    dT_dz = torch.autograd.grad(T_pred, z_col, grad_outputs=torch.ones_like(T_pred), create_graph=True)[0]
    d2T_dz2 = torch.autograd.grad(dT_dz, z_col, grad_outputs=torch.ones_like(dT_dz), create_graph=True)[0]

    K_H = 5e-7
    rho_cp = 4.18e6
    alpha = 0.07
    eta = torch.tensor(0.2, dtype=torch.float32, device=t_col.device)

    dI_dz = -batch['solar_colloc'] * (1 - alpha) * eta * torch.exp(-eta * z_col)
    heating_term = -(1.0 / rho_cp) * dI_dz
    loss_pde = torch.mean((dT_dt - K_H * d2T_dz2 - heating_term) ** 2)

    t_surface = batch['t_all'].clone().detach().requires_grad_(True)
    z_surface = torch.zeros_like(t_surface, requires_grad=True)
    T_surf_pred = model(torch.cat([t_surface, z_surface], dim=1))
    loss_surf = torch.mean((T_surf_pred - batch['T_surface_obs']) ** 2)

    z_bottom = torch.ones_like(batch['t_all']) * max_depth
    T_bottom_pred = model(torch.cat([batch['t_all'], z_bottom / max_depth], dim=1))
    loss_bot = torch.mean((T_bottom_pred - batch['T_bottom_obs']) ** 2)

    z_mld = batch['mixed_layer_depth'].clamp(0.0, max_depth)
    T_mld_pred = model(torch.cat([batch['t_all'], z_mld / max_depth], dim=1))
    loss_data = torch.mean((T_mld_pred - batch['T_surface_obs']) ** 2)

    dT_dz_surf = torch.autograd.grad(
        T_surf_pred,
        z_surface,
        grad_outputs=torch.ones_like(T_surf_pred),
        create_graph=True,
    )[0]

    wind_surf = batch['wind_norm']
    D_surf = 0.01 + 0.1 * wind_surf * torch.exp(-2.0 * z_surface)
    q_pred = -D_surf * dT_dz_surf

    K_ex = 0.5
    T_air_tensor = batch['T_air_tensor']
    q_physics = K_ex * wind_surf * (T_air_tensor - T_surf_pred)
    loss_energy_exchange = torch.mean((q_pred - q_physics) ** 2)

    Rho_pred = 1000.0 - 0.007 * (T_pred - 4.0) ** 2
    dRho_dz = torch.autograd.grad(
        Rho_pred,
        z_col,
        grad_outputs=torch.ones_like(Rho_pred),
        create_graph=True,
    )[0]
    unstable_density_violation = torch.relu(-dRho_dz)
    loss_density = torch.mean(unstable_density_violation ** 2)

    loss_total = (
        1.0 * loss_pde
        + 10.0 * loss_surf
        + 5.0 * loss_bot
        + 1.0 * loss_data
        + 2.0 * loss_energy_exchange
        + 10.0 * loss_density
    )

    return {
        'total': loss_total,
        'loss_pde': loss_pde,
        'loss_surf': loss_surf,
        'loss_bot': loss_bot,
        'loss_data': loss_data,
        'loss_energy_exchange': loss_energy_exchange,
        'loss_density': loss_density,
    }


def train_model(df, max_depth=25.0, epochs=2500, lr=1e-3, collocation_points=512, device='cpu'):
    model = LakePINN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    n_days = len(df)
    t_all = torch.tensor((df['full_doy'].values.reshape(-1, 1) / 365.0), dtype=torch.float32, device=device)
    T_surface_obs = torch.tensor(df['LST_surface_C'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    T_bottom_obs = torch.tensor(df['BottomTemp_C'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    T_air_tensor = torch.tensor(df['T_air_C'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    wind_norm = torch.tensor(df['wind_norm'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    mixed_layer_depth = torch.tensor(df['MixedLayerDepth_m'].values.reshape(-1, 1), dtype=torch.float32, device=device)
    solar_series = torch.tensor(df['Solar_J_m2'].values, dtype=torch.float32, device=device)

    for epoch in range(epochs):
        optimizer.zero_grad()

        day_index = torch.rand((collocation_points, 1), device=device) * max(n_days - 1, 1)
        day_pick = torch.round(day_index).long().clamp(0, n_days - 1).squeeze(1)
        t_colloc = (day_pick.float().unsqueeze(1) + 1.0) / 365.0
        z_colloc = torch.rand((collocation_points, 1), device=device) * max_depth
        solar_colloc = solar_series[day_pick].unsqueeze(1)

        batch = {
            't_colloc': t_colloc,
            'z_colloc': z_colloc,
            'solar_colloc': solar_colloc,
            't_all': t_all,
            'T_surface_obs': T_surface_obs,
            'T_bottom_obs': T_bottom_obs,
            'T_air_tensor': T_air_tensor,
            'wind_norm': wind_norm,
            'mixed_layer_depth': mixed_layer_depth,
        }

        losses = compute_losses(model, batch, max_depth)
        losses['total'].backward()
        optimizer.step()

        if epoch % 250 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:4d} | total={losses['total'].item():.4f} | "
                f"surf={losses['loss_surf'].item():.4f} | bot={losses['loss_bot'].item():.4f} | "
                f"pde={losses['loss_pde'].item():.4f} | energy={losses['loss_energy_exchange'].item():.4f} | "
                f"density={losses['loss_density'].item():.4f}"
            )

    return model


def predict_temperature_grid(model, n_days, max_depth=25.0, n_depth_points=150, device='cpu'):
    model.eval()
    depths = torch.linspace(0.0, max_depth, n_depth_points, device=device).reshape(-1, 1)
    z_norm = depths / max_depth
    profiles = []

    with torch.no_grad():
        for day in range(1, n_days + 1):
            t_day = torch.full_like(depths, float(day) / 365.0)
            pred = model(torch.cat([t_day, z_norm], dim=1)).cpu().numpy().flatten()
            profiles.append(pred)

    return np.array(profiles).T, depths.cpu().numpy().flatten()


def export_temperature_tables(df, temp_grid, depths, output_dir, metadata):
    records = []
    day_axis = df['full_doy'].to_numpy()
    dates = pd.to_datetime(df['Date']).to_numpy()

    for day_idx, (date_value, doy_value) in enumerate(zip(dates, day_axis)):
        month_value = pd.Timestamp(date_value).month
        for depth_idx, depth_value in enumerate(depths):
            records.append(
                {
                    'Date': pd.Timestamp(date_value).date().isoformat(),
                    'Month': month_value,
                    'DOY': int(doy_value),
                    'Depth_m': float(depth_value),
                    'Temperature_C': float(temp_grid[depth_idx, day_idx]),
                }
            )

    temp_df = pd.DataFrame.from_records(records)
    full_path = output_dir / f"{metadata['file_tag']}_temperature_depth_predictions.csv"
    july_path = output_dir / f"{metadata['file_tag']}_july_temperature_depth_predictions.csv"
    temp_df.to_csv(full_path, index=False)
    temp_df[temp_df['Month'] == 7].to_csv(july_path, index=False)
    return full_path, july_path


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
    contour_lines = ax.contour(
        day_axis,
        depths,
        temp_grid,
        levels=line_levels,
        colors='black',
        linewidths=1.1,
        alpha=0.45,
    )
    ax.clabel(contour_lines, fmt='%d', fontsize=10, inline=True)

    month_midpoints = df.groupby(df['Date'].dt.month)['full_doy'].mean()
    ax.set_xticks(month_midpoints.values)
    ax.set_xticklabels([calendar.month_abbr[month] for month in month_midpoints.index], fontsize=17)
    ax.set_xlabel('Month', fontsize=20, fontweight='bold')
    ax.set_ylabel('Depth (m)', fontsize=18, fontweight='bold')
    ax.set_title(f"Annual Water Temperature Profile of {metadata['lake_name']}", fontsize=26)
    ax.set_ylim(depths[-1], depths[0])
    ax.tick_params(axis='y', labelsize=15)

    max_depth = float(depths[-1])
    ax.text(25, max_depth * 0.10, 'Winter\nInverse\nStratification', color='blue', fontsize=22, fontweight='bold', ha='center')
    ax.text(120, max_depth * 0.88, 'Spring\nWarming', color='green', fontsize=22, fontweight='bold', ha='center')
    ax.text(210, max_depth * 0.78, 'Summer\nStratification\n(Thermocline)', color='red', fontsize=24, fontweight='bold', ha='center')
    ax.text(305, max_depth * 0.52, 'Autumn\nOverturn\n(Homothermal)', color='black', fontsize=22, fontweight='bold', ha='center')

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label('Temperature (°C)', fontsize=20)
    cbar.ax.tick_params(labelsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_monthly_heatmaps(df, temp_grid, depths, output_path, metadata):
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharey=True, constrained_layout=True)
    axes = axes.flatten()
    vmin = np.nanmin(temp_grid)
    vmax = np.nanmax(temp_grid)
    image = None

    for month in range(1, 13):
        ax = axes[month - 1]
        month_mask = df['Date'].dt.month == month
        month_temp = temp_grid[:, month_mask.values]
        if month_temp.shape[1] == 0:
            ax.set_visible(False)
            continue

        image = ax.imshow(
            month_temp,
            aspect='auto',
            origin='upper',
            extent=[1, month_temp.shape[1], depths[-1], depths[0]],
            cmap='turbo',
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(calendar.month_abbr[month])
        ax.set_xlabel('Day of month')
        if (month - 1) % 4 == 0:
            ax.set_ylabel('Depth (m)')

    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), shrink=0.92, label='Temperature (°C)')
    fig.suptitle(f"{metadata['lake_name']} {metadata['year']} Monthly Temperature-Depth Heatmaps", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='Train the lake PINN and draw annual heatmaps.')
    parser.add_argument('--era5', default=DEFAULT_ERA5_PATH, help='Path to ERA5 daily CSV')
    parser.add_argument('--lst', default=DEFAULT_LST_PATH, help='Path to MODIS LST CSV')
    parser.add_argument('--max-depth', type=float, default=25.0, help='Plot depth in meters')
    parser.add_argument('--depth-points', type=int, default=150, help='Number of depth samples')
    parser.add_argument('--epochs', type=int, default=2500, help='Training epochs')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output-dir', default=str(PROJECT_DIR / 'outputs'), help='Directory for saved figures')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, metadata = load_training_frame(args.era5, args.lst)
    model = train_model(df, max_depth=args.max_depth, epochs=args.epochs, lr=args.lr)
    temp_grid, depths = predict_temperature_grid(
        model,
        n_days=len(df),
        max_depth=args.max_depth,
        n_depth_points=args.depth_points,
    )

    year_path = output_dir / f"{metadata['file_tag']}_year_heatmap.png"
    monthly_path = output_dir / f"{metadata['file_tag']}_monthly_heatmaps.png"
    full_csv_path, july_csv_path = export_temperature_tables(df, temp_grid, depths, output_dir, metadata)
    plot_year_heatmap(df, temp_grid, depths, year_path, metadata)
    plot_monthly_heatmaps(df, temp_grid, depths, monthly_path, metadata)

    print(f"Saved annual heatmap to: {year_path}")
    print(f"Saved monthly heatmaps to: {monthly_path}")
    print(f"Saved full prediction table to: {full_csv_path}")
    print(f"Saved July prediction table to: {july_csv_path}")


if __name__ == '__main__':
    main()
