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
        raw_value = input(f"请输入{label}文件路径: ")
        if not raw_value:
            print(f"{label}文件路径不能为空，请重新输入。")
            continue

        path = normalize_input_path(raw_value)
        if path.exists() and path.is_file():
            return path.resolve()

        print(f"{label}文件不存在: {path}")
        print("请检查路径后重新输入。")


def prompt_for_existing_path(label: str) -> Path:
    try:
        enable_high_dpi()
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        root.tk.call("tk", "scaling", 1.0)
        file_path = filedialog.askopenfilename(
            title=f"请选择{label}文件",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        root.destroy()
    except Exception:
        file_path = ""

    if file_path:
        path = Path(file_path)
        if path.exists() and path.is_file():
            print(f"已选择{label}文件: {path}")
            return path.resolve()

    print(f"未通过弹窗选择{label}文件，切换为终端输入。")
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


def resolve_month(month_arg: int | None) -> int | None:
    if month_arg is not None:
        if 1 <= month_arg <= 12:
            return month_arg
        raise ValueError("--month must be between 1 and 12.")

    while True:
        raw_value = input("请输入月份(1-12)，直接回车表示全年: ").strip()
        if not raw_value:
            return None

        try:
            month = int(raw_value)
        except ValueError:
            print("月份必须是1到12的整数，请重新输入。")
            continue

        if 1 <= month <= 12:
            return month

        print("月份必须在1到12之间，请重新输入。")


def infer_lake_name_from_path(path: Path) -> str:
    text = path.stem.replace("-", " ").replace("_", " ")
    text = re.sub(r"\b(19|20)\d{2}\b", " ", text)
    tokens_to_remove = [
        "temperature", "temperatures", "profile", "profiles", "records", "record",
        "observed", "simulate", "simulated", "prediction", "predictions",
        "daily", "hourly", "era5", "modis", "lst", "depth", "lake",
    ]
    for token in tokens_to_remove:
        text = re.sub(rf"\b{token}\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text.title() if text else "Lake"


def infer_year_label(df: pd.DataFrame) -> str | None:
    if "Date" not in df.columns or df["Date"].dropna().empty:
        return None

    years = sorted(df["Date"].dropna().dt.year.unique().tolist())
    if not years:
        return None
    if len(years) == 1:
        return str(years[0])
    return f"{years[0]}-{years[-1]}"


def detect_column(columns: list[str], candidates: list[str], explicit: str | None, label: str) -> str:
    if explicit:
        if explicit in columns:
            return explicit
        raise ValueError(f"{label} column '{explicit}' not found. Available columns: {columns}")

    lowered = {col.lower(): col for col in columns}
    for name in candidates:
        if name in columns:
            return name
        if name.lower() in lowered:
            return lowered[name.lower()]

    raise ValueError(f"Could not detect {label} column. Available columns: {columns}")


def detect_date_column(columns: list[str], explicit: str | None) -> str | None:
    if explicit:
        if explicit in columns:
            return explicit
        raise ValueError(f"Date column '{explicit}' not found. Available columns: {columns}")

    lowered = {col.lower(): col for col in columns}
    for name in ["Date", "datetime", "Datetime", "date", "time", "Time", "timestamp", "Timestamp"]:
        if name in columns:
            return name
        if name.lower() in lowered:
            return lowered[name.lower()]
    return None


def depth_from_column(column_name: str) -> float:
    match = re.search(r"Temp_(-?\d+(?:\.\d+)?)m", column_name)
    if not match:
        raise ValueError(f"Unexpected temperature column: {column_name}")
    return float(match.group(1))


def detect_long_temperature_column(columns: list[str], explicit: str | None) -> str:
    candidates = [
        "Temperature_C",
        "Temperature_degCelsius",
        "Temperature",
        "Temp_C",
        "Temp",
        "temp",
    ]
    return detect_column(columns, candidates, explicit, "temperature")


def load_profile_data(
    csv_path: Path,
    depth_col_arg: str | None,
    temp_col_arg: str | None,
    date_col_arg: str | None,
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    columns = df.columns.tolist()
    date_col = detect_date_column(columns, date_col_arg)
    wide_temp_columns = [col for col in columns if col.startswith("Temp_")]

    if wide_temp_columns:
        if date_col is None:
            raise ValueError(
                "Wide-format input requires a date column plus one or more Temp_* columns."
            )
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col]).copy()
        if df.empty:
            raise ValueError(f"No valid date rows remained after cleaning: {csv_path}")

        for column in wide_temp_columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

        long_df = df.melt(
            id_vars=[date_col],
            value_vars=wide_temp_columns,
            var_name="DepthColumn",
            value_name="Temperature_C",
        )
        long_df["Depth_m"] = long_df["DepthColumn"].map(depth_from_column)
        long_df = long_df.rename(columns={date_col: "Date"})
        long_df["Temperature_C"] = pd.to_numeric(long_df["Temperature_C"], errors="coerce")
        long_df = long_df.dropna(subset=["Depth_m", "Temperature_C"]).copy()
        return long_df[["Date", "Depth_m", "Temperature_C"]].reset_index(drop=True)

    depth_col = detect_column(columns, ["Depth_m", "Depth", "depth_m", "depth"], depth_col_arg, "depth")
    temp_col = detect_long_temperature_column(columns, temp_col_arg)

    normalized = pd.DataFrame(
        {
            "Depth_m": pd.to_numeric(df[depth_col], errors="coerce"),
            "Temperature_C": pd.to_numeric(df[temp_col], errors="coerce"),
        }
    )
    if date_col is not None:
        normalized["Date"] = pd.to_datetime(df[date_col], errors="coerce")

    normalized = normalized.dropna(subset=["Depth_m", "Temperature_C"]).copy()
    if normalized.empty:
        raise ValueError(f"No valid rows remained after cleaning: {csv_path}")

    if "Date" in normalized.columns:
        normalized = normalized.dropna(subset=["Date"]).copy()
        if normalized.empty:
            raise ValueError(f"No valid date rows remained after cleaning: {csv_path}")

    ordered_columns = ["Date", "Depth_m", "Temperature_C"] if "Date" in normalized.columns else ["Depth_m", "Temperature_C"]
    return normalized[ordered_columns].reset_index(drop=True)


def load_mean_profile(
    csv_path: Path,
    depth_col_arg: str | None,
    temp_col_arg: str | None,
    date_col_arg: str | None,
    month: int | None,
) -> pd.DataFrame:
    df = load_profile_data(csv_path, depth_col_arg, temp_col_arg, date_col_arg)

    if month is not None:
        if "Date" not in df.columns:
            raise ValueError("Month filtering was requested but no date column was found.")
        df = df[df["Date"].dt.month == month].copy()
        if df.empty:
            raise ValueError(f"No rows remained after applying month={month}: {csv_path}")

    mean_profile = (
        df.groupby("Depth_m", as_index=False)["Temperature_C"]
        .mean()
        .sort_values("Depth_m")
        .reset_index(drop=True)
    )
    return mean_profile


def align_profiles(sim_mean: pd.DataFrame, obs_mean: pd.DataFrame) -> pd.DataFrame:
    obs_depth = obs_mean["Depth_m"].to_numpy()
    sim_interp = np.interp(
        obs_depth,
        sim_mean["Depth_m"].to_numpy(),
        sim_mean["Temperature_C"].to_numpy(),
    )
    obs_temp = obs_mean["Temperature_C"].to_numpy()
    err = sim_interp - obs_temp

    aligned = pd.DataFrame(
        {
            "Depth_m": obs_depth,
            "ObservedMean_C": obs_temp,
            "SimulatedMean_C": sim_interp,
            "Error_C": err,
        }
    )
    return aligned


def compute_metrics(aligned: pd.DataFrame) -> tuple[float, float, float, float]:
    err = aligned["Error_C"].to_numpy()
    obs = aligned["ObservedMean_C"].to_numpy()
    sim = aligned["SimulatedMean_C"].to_numpy()

    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    cor = float(np.corrcoef(sim, obs)[0, 1]) if len(obs) > 1 else float("nan")
    return rmse, mae, bias, cor


def estimate_thermocline_depth(depth: np.ndarray, temperature: np.ndarray) -> tuple[float, float]:
    gradient = np.gradient(temperature, depth)
    return float(depth[np.argmin(gradient)]), float(np.min(gradient))


def plot_comparison(aligned: pd.DataFrame, metrics: tuple[float, float, float, float], lake_name: str, period_label: str, out_png: Path) -> None:
    rmse, mae, bias, cor = metrics

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(15, 8.5),
        gridspec_kw={"width_ratios": [1.25, 0.95]},
        constrained_layout=True,
    )
    ax_profile, ax_error = axes

    depth = aligned["Depth_m"].to_numpy()
    obs = aligned["ObservedMean_C"].to_numpy()
    sim = aligned["SimulatedMean_C"].to_numpy()
    err = aligned["Error_C"].to_numpy()

    dense_depth = np.linspace(depth.min(), depth.max(), 300)
    obs_dense = np.interp(dense_depth, depth, obs)
    sim_dense = np.interp(dense_depth, depth, sim)

    obs_therm_depth, _ = estimate_thermocline_depth(depth, obs)
    sim_therm_depth, _ = estimate_thermocline_depth(depth, sim)

    ax_profile.fill_betweenx(depth, obs, sim, color="#f4b183", alpha=0.28, label="Gap between curves")
    ax_profile.plot(obs_dense, dense_depth, color="#111111", linewidth=3.2, label=f"Observed {period_label}")
    ax_profile.plot(sim_dense, dense_depth, color="#c62828", linewidth=3.2, label=f"Simulated {period_label}")
    ax_profile.scatter(obs, depth, color="#111111", s=44, zorder=3)
    ax_profile.scatter(sim, depth, color="#c62828", s=44, zorder=3)
    ax_profile.axhline(obs_therm_depth, color="#111111", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_profile.axhline(sim_therm_depth, color="#c62828", linestyle="--", linewidth=1.2, alpha=0.7)
    ax_profile.set_ylim(depth.max(), depth.min())
    ax_profile.set_xlabel("Temperature (deg C)", fontsize=16, fontweight="bold")
    ax_profile.set_ylabel("Depth (m)", fontsize=16, fontweight="bold")
    ax_profile.set_title(f"{period_label.capitalize()} Temperature Profile", fontsize=18, fontweight="bold")
    ax_profile.tick_params(axis="both", labelsize=12)
    ax_profile.grid(alpha=0.16, color="black", linewidth=0.8)
    ax_profile.legend(loc="lower right", fontsize=10.5, frameon=True)
    ax_profile.text(
        0.03,
        0.04,
        (
            f"Observed thermocline ~ {obs_therm_depth:.1f} m\n"
            f"Simulated thermocline ~ {sim_therm_depth:.1f} m"
        ),
        transform=ax_profile.transAxes,
        fontsize=11,
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none", "boxstyle": "round,pad=0.4"},
    )

    colors = np.where(err >= 0, "#1f77b4", "#d62728")
    ax_error.barh(depth, err, color=colors, alpha=0.82, height=0.32, edgecolor="none")
    ax_error.plot(err, depth, color="#333333", linewidth=1.5, alpha=0.65)
    ax_error.scatter(err, depth, c=colors, s=46, edgecolors="white", linewidths=0.6, zorder=3)
    ax_error.axvline(0.0, color="black", linewidth=1.25)
    ax_error.set_ylim(depth.max(), depth.min())
    ax_error.set_xlabel("Simulated - Observed (deg C)", fontsize=16, fontweight="bold")
    ax_error.set_title("Depth-wise Error", fontsize=18, fontweight="bold")
    ax_error.tick_params(axis="both", labelsize=12)
    ax_error.grid(alpha=0.16, color="black", linewidth=0.8)

    metrics_text = (
        f"RMSE = {rmse:.2f} deg C\n"
        f"MAE = {mae:.2f} deg C\n"
        f"Bias = {bias:.2f} deg C\n"
        f"Correlation = {cor:.2f}"
    )
    ax_error.text(
        0.05,
        0.05,
        metrics_text,
        transform=ax_error.transAxes,
        fontsize=12,
        bbox={"facecolor": "white", "alpha": 0.84, "edgecolor": "none", "boxstyle": "round,pad=0.4"},
    )

    fig.suptitle(f"Observed vs Simulated Profile of {lake_name}", fontsize=22, fontweight="bold")
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def build_period_labels(month: int | None) -> tuple[str, str]:
    if month is None:
        return "all_time_mean", "all-time mean"
    return f"month_{month:02d}", f"{calendar.month_name[month]} mean"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare observed and simulated temperature profiles for any lake CSV files."
    )
    parser.add_argument("--sim-input", default=None, help="Path to simulated profile CSV.")
    parser.add_argument("--obs-input", default=None, help="Path to observed profile CSV.")
    parser.add_argument("--lake-name", default=None, help="Optional lake name used in titles and output names.")
    parser.add_argument("--output-dir", default=None, help="Directory for output files.")
    parser.add_argument("--month", type=int, default=None, help="Optional month filter (1-12).")

    parser.add_argument("--sim-depth-col", default=None, help="Depth column name in simulated CSV.")
    parser.add_argument("--sim-temp-col", default=None, help="Temperature column name in simulated CSV.")
    parser.add_argument("--sim-date-col", default=None, help="Date column name in simulated CSV (used with --month).")
    parser.add_argument("--obs-depth-col", default=None, help="Depth column name in observed CSV.")
    parser.add_argument("--obs-temp-col", default=None, help="Temperature column name in observed CSV.")
    parser.add_argument("--obs-date-col", default=None, help="Date column name in observed CSV (used with --month).")

    args = parser.parse_args()

    if args.sim_input:
        sim_input = resolve_input_file(normalize_input_path(args.sim_input))
    else:
        sim_input = prompt_for_existing_path("模拟")

    if args.obs_input:
        obs_input = resolve_input_file(normalize_input_path(args.obs_input))
    else:
        obs_input = prompt_for_existing_path("观测")

    month = resolve_month(args.month)

    sim_profile_data = load_profile_data(sim_input, args.sim_depth_col, args.sim_temp_col, args.sim_date_col)
    obs_profile_data = load_profile_data(obs_input, args.obs_depth_col, args.obs_temp_col, args.obs_date_col)
    sim_mean = load_mean_profile(sim_input, args.sim_depth_col, args.sim_temp_col, args.sim_date_col, month)
    obs_mean = load_mean_profile(obs_input, args.obs_depth_col, args.obs_temp_col, args.obs_date_col, month)
    aligned = align_profiles(sim_mean, obs_mean)

    period_tag, period_label = build_period_labels(month)
    lake_name = args.lake_name.strip() if args.lake_name else infer_lake_name_from_path(obs_input)
    year_label = infer_year_label(obs_profile_data) or infer_year_label(sim_profile_data)
    file_tag = f"{slugify(lake_name)}_{year_label}" if year_label else slugify(lake_name)

    output_dir = resolve_output_dir(args.output_dir, sim_input.parent / "outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    out_csv = output_dir / f"{file_tag}_observed_vs_simulated_{period_tag}_aligned.csv"
    out_png = output_dir / f"{file_tag}_observed_vs_simulated_{period_tag}.png"

    aligned.to_csv(out_csv, index=False)
    metrics = compute_metrics(aligned)
    plot_comparison(aligned, metrics, lake_name, period_label, out_png)

    print(f"Saved aligned comparison table to: {out_csv}")
    print(f"Saved comparison figure to: {out_png}")


if __name__ == "__main__":
    main()
