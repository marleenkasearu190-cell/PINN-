import argparse
import calendar
import ctypes
from pathlib import Path
import re
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 220


def slugify(text: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip())
    return slug.strip("_") or "lake"


def normalize_input_path(raw_value: str) -> Path:
    cleaned = raw_value.strip()
    if cleaned.startswith("& "):
        cleaned = cleaned[2:].strip()
    cleaned = cleaned.strip('"').strip("'")
    return Path(cleaned).expanduser()


def enable_high_dpi() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass


def ask_path_in_terminal(label: str) -> Path:
    while True:
        raw_value = input(f"请输入 {label} 文件路径: ")
        if not raw_value:
            print(f"{label} 文件路径不能为空，请重新输入。")
            continue

        path = normalize_input_path(raw_value)
        if path.exists() and path.is_file():
            return path.resolve()

        print(f"{label} 文件不存在: {path}")
        print("请检查路径后重新输入。")


def prompt_for_existing_path(label: str) -> Path:
    try:
        enable_high_dpi()
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.tk.call("tk", "scaling", 1.0)
        file_path = filedialog.askopenfilename(
            title=f"请选择 {label} 文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
    except Exception:
        file_path = ""

    if file_path:
        path = Path(file_path)
        if path.exists() and path.is_file():
            print(f"已选择 {label} 文件: {path}")
            return path.resolve()

    print(f"未通过弹窗选择 {label} 文件，切换为终端输入。")
    return ask_path_in_terminal(label)


def resolve_input_file(input_file: Path) -> Path:
    input_file = input_file.expanduser()
    if input_file.exists() and input_file.is_file():
        return input_file.resolve()
    raise FileNotFoundError(f"Input file not found: {input_file}")


def resolve_output_dir(output_dir_arg: str | None, default_dir: Path) -> Path:
    if output_dir_arg:
        return normalize_input_path(output_dir_arg).resolve()

    raw_value = input(f"请输入保存目录，直接回车使用默认目录 [{default_dir}]: ").strip()
    if not raw_value:
        return default_dir.resolve()

    return normalize_input_path(raw_value).resolve()


def infer_lake_name_from_path(input_file: Path) -> str:
    text = input_file.stem.replace("-", " ").replace("_", " ")
    text = re.sub(r"\b(19|20)\d{2}\b", " ", text)
    tokens_to_remove = [
        "temperature", "temperatures", "profile", "profiles", "best", "year",
        "validation", "sonde", "observed", "prediction", "predictions",
        "daily", "hourly", "era5", "modis", "lst", "lake",
    ]
    for token in tokens_to_remove:
        text = re.sub(rf"\b{token}\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.title() if text else "Lake"


def infer_metadata(df: pd.DataFrame, input_file: Path, lake_name_arg: str | None) -> dict[str, str]:
    year_label = infer_year_label(df)
    lake_name = lake_name_arg.strip() if lake_name_arg else infer_lake_name_from_path(input_file)
    return {
        "lake_name": lake_name,
        "year_label": year_label,
        "file_tag": f"{slugify(lake_name)}_{year_label}",
    }


def depth_from_column(column_name: str) -> float:
    match = re.search(r"Temp_(-?\d+(?:\.\d+)?)m", column_name)
    if not match:
        raise ValueError(f"Unexpected temperature column: {column_name}")
    return float(match.group(1))


def detect_long_temperature_column(columns: list[str]) -> str:
    candidates = [
        "Temperature_C",
        "Temperature_degCelsius",
        "Temperature",
        "Temp_C",
        "Temp",
    ]
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(
        "Could not find a temperature column in long-format data. "
        "Expected one of: Temperature_C, Temperature_degCelsius, Temperature, Temp_C, Temp"
    )


def load_profile_data(input_file: Path) -> pd.DataFrame:
    df = pd.read_csv(input_file)
    if "Date" not in df.columns:
        raise ValueError("Input file must contain a 'Date' column.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").copy()

    wide_temp_columns = [col for col in df.columns if col.startswith("Temp_")]
    if wide_temp_columns:
        for column in wide_temp_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
        grouped = df.groupby("Date", as_index=False)[wide_temp_columns].mean()
        return grouped.sort_values("Date").reset_index(drop=True)

    if "Depth_m" in df.columns:
        df["Depth_m"] = pd.to_numeric(df["Depth_m"], errors="coerce")
        temp_column = detect_long_temperature_column(df.columns.tolist())
        df[temp_column] = pd.to_numeric(df[temp_column], errors="coerce")
        df = df.dropna(subset=["Depth_m", temp_column]).copy()
        if df.empty:
            raise ValueError("No valid depth-temperature rows remained after cleaning.")

        pivot = (
            df.groupby(["Date", "Depth_m"], as_index=False)[temp_column]
            .mean()
            .pivot(index="Date", columns="Depth_m", values=temp_column)
            .sort_index(axis=1)
        )
        pivot.columns = [f"Temp_{depth:g}m" for depth in pivot.columns]
        pivot = pivot.reset_index()
        return pivot.sort_values("Date").reset_index(drop=True)

    raise ValueError(
        "Unsupported input format. Provide either wide-format data with Temp_* columns "
        "or long-format data with Date, Depth_m, and temperature columns."
    )


def infer_year_label(df: pd.DataFrame) -> str:
    years = sorted(df["Date"].dt.year.unique().tolist())
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}-{years[-1]}"


def build_temperature_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    temp_columns = [col for col in df.columns if col.startswith("Temp_")]
    depths = np.array([depth_from_column(col) for col in temp_columns], dtype=float)
    day_axis = df["Date"].dt.dayofyear.to_numpy(dtype=float)
    temp_grid = df[temp_columns].to_numpy(dtype=float).T
    return temp_grid, depths, day_axis


def plot_heatmap(df: pd.DataFrame, lake_name: str, year_label: str, output_path: Path) -> None:
    temp_grid, depths, day_axis = build_temperature_grid(df)
    masked_grid = np.ma.masked_invalid(temp_grid)

    fig, ax = plt.subplots(figsize=(14, 8), constrained_layout=True)
    vmin = float(np.nanmin(temp_grid))
    vmax = float(np.nanmax(temp_grid))
    filled_levels = np.linspace(vmin, vmax, 28)
    line_levels = np.arange(np.floor(vmin / 4.0) * 4.0, np.ceil(vmax / 4.0) * 4.0 + 0.1, 4.0)
    if line_levels.size < 2:
        line_levels = np.linspace(vmin, vmax, 6)

    image = ax.contourf(
        day_axis,
        depths,
        masked_grid,
        levels=filled_levels,
        cmap="RdYlBu_r",
        extend="both",
    )
    contour_lines = ax.contour(
        day_axis,
        depths,
        masked_grid,
        levels=line_levels,
        colors="black",
        linewidths=1.0,
        alpha=0.42,
    )
    ax.clabel(contour_lines, fmt="%d", fontsize=10, inline=True)

    month_midpoints = df.groupby(df["Date"].dt.month)["Date"].apply(lambda x: x.dt.dayofyear.mean())
    ax.set_xticks(month_midpoints.values)
    ax.set_xticklabels([calendar.month_abbr[m] for m in month_midpoints.index], fontsize=17)
    ax.set_xlabel("Month", fontsize=20, fontweight="bold")
    ax.set_ylabel("Depth (m)", fontsize=16)
    ax.set_title(f"Annual Water Temperature Profile of {lake_name}", fontsize=26)
    ax.set_ylim(depths[-1], depths[0])
    ax.tick_params(axis="y", labelsize=15)

    max_depth = float(depths[-1])
    ax.text(30, max_depth * 0.12, "Winter\nInverse\nStratification", color="blue", fontsize=20, fontweight="bold", ha="center")
    ax.text(122, max_depth * 0.82, "Spring\nWarming", color="green", fontsize=20, fontweight="bold", ha="center")
    ax.text(212, max_depth * 0.72, "Summer\nStratification\n(Thermocline)", color="red", fontsize=22, fontweight="bold", ha="center")
    ax.text(302, max_depth * 0.46, "Autumn\nOverturn", color="black", fontsize=20, fontweight="bold", ha="center")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Temperature (C)", fontsize=20)
    cbar.ax.tick_params(labelsize=14)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_heatmaps(df: pd.DataFrame, lake_name: str, year_label: str, output_path: Path) -> None:
    temp_columns = [col for col in df.columns if col.startswith("Temp_")]
    depths = np.array([depth_from_column(col) for col in temp_columns], dtype=float)
    fig, axes = plt.subplots(3, 4, figsize=(16, 12), sharey=True, constrained_layout=True)
    axes = axes.flatten()

    all_values = df[temp_columns].to_numpy(dtype=float)
    vmin = float(np.nanmin(all_values))
    vmax = float(np.nanmax(all_values))
    image = None

    for month in range(1, 13):
        ax = axes[month - 1]
        month_df = df[df["Date"].dt.month == month].copy()
        if month_df.empty:
            ax.set_visible(False)
            continue

        days_in_month = int(month_df["Date"].dt.days_in_month.iloc[0])
        month_grid = np.full((len(temp_columns), days_in_month), np.nan, dtype=float)
        for _, row in month_df.iterrows():
            day = int(row["Date"].day) - 1
            month_grid[:, day] = row[temp_columns].to_numpy(dtype=float)

        month_masked = np.ma.masked_invalid(month_grid)
        image = ax.imshow(
            month_masked,
            aspect="auto",
            origin="upper",
            extent=[1, days_in_month, depths[-1], depths[0]],
            cmap="turbo",
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(calendar.month_abbr[month], fontsize=12, fontweight="bold")
        ax.set_xlabel("Day of month")
        if (month - 1) % 4 == 0:
            ax.set_ylabel("Depth (m)")

    if image is not None:
        fig.colorbar(image, ax=axes.tolist(), shrink=0.92, label="Temperature (C)")
    fig.suptitle(f"{lake_name} {year_label} Monthly Temperature-Depth Heatmaps", fontsize=14)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot annual and monthly temperature-depth heatmaps for any lake from a yearly CSV."
    )
    parser.add_argument("--input", default=None, help="Path to the yearly lake temperature CSV.")
    parser.add_argument("--lake-name", default=None, help="Optional lake name used in titles and output names.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for output figures. Defaults to an 'outputs' folder next to the input file.",
    )
    args = parser.parse_args()

    if args.input:
        input_file = resolve_input_file(normalize_input_path(args.input))
    else:
        input_file = prompt_for_existing_path("温度剖面 CSV")

    df = load_profile_data(input_file)
    metadata = infer_metadata(df, input_file, args.lake_name)

    output_dir = resolve_output_dir(args.output_dir, input_file.parent / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    out_heatmap = output_dir / f"{metadata['file_tag']}_profile_heatmap.png"
    out_monthly = output_dir / f"{metadata['file_tag']}_monthly_heatmaps.png"
    plot_heatmap(df, metadata["lake_name"], metadata["year_label"], out_heatmap)
    plot_monthly_heatmaps(df, metadata["lake_name"], metadata["year_label"], out_monthly)

    print("Visualization complete:")
    print(out_heatmap)
    print(out_monthly)


if __name__ == "__main__":
    main()
