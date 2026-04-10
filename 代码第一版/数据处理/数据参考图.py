import argparse
import ctypes
import re
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 200


MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SEASON_MAP = {
    12: "Winter", 1: "Winter", 2: "Winter",
    3: "Spring", 4: "Spring", 5: "Spring",
    6: "Summer", 7: "Summer", 8: "Summer",
    9: "Autumn", 10: "Autumn", 11: "Autumn",
}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Autumn"]
QA_GOOD = "LST produced, good quality, not necessary to examine more detailed QA"
QA_OTHER = "LST produced, other quality, recommend examination of more detailed QA"


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


def resolve_existing_file(path_arg: str | None, label: str) -> Path:
    if path_arg:
        path = normalize_input_path(path_arg)
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"{label} file not found: {path}")
        return path.resolve()
    return prompt_for_existing_path(label)


def prompt_for_output_dir(default_dir: Path) -> Path:
    raw_value = input(f"请输入保存目录，直接回车使用默认目录 [{default_dir}]: ").strip()
    if not raw_value:
        return default_dir.resolve()
    return normalize_input_path(raw_value).resolve()


def sanitize_name(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def infer_lake_name_from_paths(daily_path: Path, lst_path: Path) -> str:
    candidates = [
        lst_path.stem,
        daily_path.stem,
    ]
    tokens_to_remove = [
        "era5", "daily", "hourly", "results", "result", "mod11a1", "061", "modis",
        "lst", "lake", "temperature", "data",
    ]
    for candidate in candidates:
        text = candidate.replace("-", " ").replace("_", " ")
        text = re.sub(r"\b(19|20)\d{2}\b", " ", text)
        for token in tokens_to_remove:
            text = re.sub(rf"\b{token}\b", " ", text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        if text:
            return text.title()
    return "Lake"


def infer_metadata(
    daily: pd.DataFrame,
    daily_path: Path,
    lst_path: Path,
    lake_name_arg: str | None,
) -> dict[str, str | int]:
    year = int(daily["Date"].dt.year.mode().iloc[0])
    lake_name = lake_name_arg.strip() if lake_name_arg else infer_lake_name_from_paths(daily_path, lst_path)
    file_tag = f"{sanitize_name(lake_name) or 'lake'}_{year}"
    return {"lake_name": lake_name, "year": year, "file_tag": file_tag}


def infer_output_dir(
    daily_path: Path,
    hourly_path: Path,
    lst_path: Path,
    output_dir_arg: str | None,
) -> Path:
    if output_dir_arg:
        return Path(output_dir_arg).expanduser().resolve()
    common_parent = daily_path.parent
    if hourly_path.parent == common_parent and lst_path.parent == common_parent:
        return common_parent / "outputs"
    return daily_path.parent / "outputs"


def load_data(daily_path: Path, hourly_path: Path, lst_path: Path):
    daily = pd.read_csv(daily_path, parse_dates=["Date"])
    hourly = pd.read_csv(hourly_path, parse_dates=["Date"])
    lst = pd.read_csv(lst_path, parse_dates=["Date"])

    daily = daily.sort_values("Date").copy()
    hourly = hourly.sort_values("Date").copy()
    lst = lst.sort_values("Date").copy()

    year = int(daily["Date"].dt.year.mode().iloc[0])
    lst = lst[(lst["Date"] >= f"{year}-01-01") & (lst["Date"] <= f"{year}-12-31")].copy()
    lst["LST_K"] = pd.to_numeric(lst["MOD11A1_061_LST_Day_1km"], errors="coerce")
    lst["LST_C"] = lst["LST_K"] - 273.15
    lst["is_valid"] = lst["LST_K"] > 0

    lst_valid = lst[lst["is_valid"]].copy()
    qc_column = "MOD11A1_061_QC_Day_MODLAND_Description"
    if qc_column in lst_valid.columns:
        lst_good = lst_valid[lst_valid[qc_column] == QA_GOOD].copy()
        lst_other = lst_valid[lst_valid[qc_column] == QA_OTHER].copy()
    else:
        lst_good = lst_valid.copy()
        lst_other = lst_valid.iloc[0:0].copy()

    merged = daily.merge(lst_valid[["Date", "LST_C"]], on="Date", how="left")

    daily["Month"] = daily["Date"].dt.month
    daily["MonthLabel"] = pd.Categorical(
        [MONTH_LABELS[m - 1] for m in daily["Month"]],
        categories=MONTH_LABELS,
        ordered=True,
    )
    daily["Season"] = pd.Categorical(daily["Month"].map(SEASON_MAP), categories=SEASON_ORDER, ordered=True)

    valid = merged.dropna(subset=["LST_C"]).copy()
    valid["Month"] = valid["Date"].dt.month
    valid["MonthLabel"] = pd.Categorical(
        [MONTH_LABELS[m - 1] for m in valid["Month"]],
        categories=MONTH_LABELS,
        ordered=True,
    )
    valid["Season"] = pd.Categorical(valid["Month"].map(SEASON_MAP), categories=SEASON_ORDER, ordered=True)

    return daily, hourly, lst, lst_good, lst_other, merged, valid


def build_output_paths(output_dir: Path, metadata: dict[str, str | int]) -> dict[str, Path]:
    tag = str(metadata["file_tag"])
    return {
        "overview": output_dir / f"{tag}_overview.png",
        "temp_compare": output_dir / f"{tag}_temperature_comparison.png",
        "hourly_heatmap": output_dir / f"{tag}_hourly_heatmap.png",
        "monthly": output_dir / f"{tag}_monthly_seasonal.png",
        "corr": output_dir / f"{tag}_correlation_matrix.png",
    }


def plot_overview(daily, lst_good, lst_other, metadata, output_path: Path):
    fig, axes = plt.subplots(4, 1, figsize=(15, 13), sharex=True)

    axes[0].plot(daily["Date"], daily["t2m_C"], label="ERA5 2m air temperature", color="#d95f02", linewidth=1.6)
    axes[0].plot(daily["Date"], daily["lblt_C"], label="ERA5 lake bottom temperature", color="#1b9e77", linewidth=1.6)
    axes[0].scatter(lst_good["Date"], lst_good["LST_C"], label="MODIS LST good QA", color="#1f78b4", s=18, alpha=0.85)
    if not lst_other.empty:
        axes[0].scatter(lst_other["Date"], lst_other["LST_C"], label="MODIS LST other QA", color="#7570b3", s=16, alpha=0.55)
    axes[0].set_ylabel("Temperature (C)")
    axes[0].set_title(f"{metadata['lake_name']} {metadata['year']}: ERA5 and MODIS overview")
    axes[0].legend(loc="upper right", ncol=2, fontsize=8)

    axes[1].plot(daily["Date"], daily["wind_norm_m_per_s"], color="#386cb0", linewidth=1.4)
    axes[1].fill_between(daily["Date"], daily["wind_norm_m_per_s"], color="#386cb0", alpha=0.16)
    axes[1].set_ylabel("Wind speed (m/s)")
    axes[1].set_title("10 m wind speed magnitude")

    axes[2].plot(daily["Date"], daily["Is_J_per_m2"], color="#e6ab02", linewidth=1.4)
    axes[2].fill_between(daily["Date"], daily["Is_J_per_m2"], color="#e6ab02", alpha=0.2)
    axes[2].set_ylabel("J/m2")
    axes[2].set_title("Surface solar radiation downwards")

    axes[3].plot(daily["Date"], daily["lmld_m"], color="#66a61e", linewidth=1.4)
    axes[3].fill_between(daily["Date"], daily["lmld_m"], color="#66a61e", alpha=0.2)
    axes[3].set_ylabel("Depth (m)")
    axes[3].set_title("Lake mixed layer depth")
    axes[3].set_xlabel("Date")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_temperature_comparison(merged, output_path: Path):
    valid = merged.dropna(subset=["LST_C"]).copy()
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    axes[0].plot(merged["Date"], merged["t2m_C"], label="ERA5 2m air temperature", color="#d95f02", linewidth=1.5)
    axes[0].plot(merged["Date"], merged["lblt_C"], label="ERA5 lake bottom temperature", color="#1b9e77", linewidth=1.5)
    axes[0].scatter(valid["Date"], valid["LST_C"], label="MODIS daytime LST", color="#1f78b4", s=20, alpha=0.9)
    axes[0].set_ylabel("Temperature (C)")
    axes[0].set_title("Temperature comparison on days with valid MODIS LST")
    axes[0].legend(loc="upper right")

    axes[1].scatter(valid["t2m_C"], valid["LST_C"], label="LST vs ERA5 air temperature", color="#d95f02", s=22, alpha=0.8)
    axes[1].scatter(valid["lblt_C"], valid["LST_C"], label="LST vs ERA5 bottom temperature", color="#1b9e77", s=22, alpha=0.7)
    if len(valid) > 1:
        corr_air = valid[["t2m_C", "LST_C"]].corr().iloc[0, 1]
        corr_bottom = valid[["lblt_C", "LST_C"]].corr().iloc[0, 1]
        axes[1].text(
            0.02,
            0.98,
            f"corr(LST, t2m) = {corr_air:.2f}\ncorr(LST, lblt) = {corr_bottom:.2f}",
            transform=axes[1].transAxes,
            va="top",
            fontsize=10,
        )
    axes[1].set_xlabel("ERA5 temperature (C)")
    axes[1].set_ylabel("MODIS LST (C)")
    axes[1].set_title("Pointwise comparison on valid LST days")
    axes[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_hourly_heatmap(hourly, output_path: Path):
    hourly = hourly.copy()
    hourly["day_of_year"] = hourly["Date"].dt.dayofyear
    hourly["hour"] = hourly["Date"].dt.hour

    t2m_grid = hourly.pivot(index="hour", columns="day_of_year", values="t2m_C")
    wind_grid = hourly.pivot(index="hour", columns="day_of_year", values="wind_norm_m_per_s")

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    sns.heatmap(t2m_grid, ax=axes[0], cmap="coolwarm", cbar_kws={"label": "C"})
    axes[0].set_title("Hourly 2 m air temperature heatmap")
    axes[0].set_ylabel("Hour of day")

    sns.heatmap(wind_grid, ax=axes[1], cmap="YlGnBu", cbar_kws={"label": "m/s"})
    axes[1].set_title("Hourly 10 m wind speed heatmap")
    axes[1].set_ylabel("Hour of day")
    axes[1].set_xlabel("Day of year")

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_monthly_seasonal(daily, valid, output_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    monthly = daily.groupby("MonthLabel", observed=False)[["t2m_C", "lblt_C", "wind_norm_m_per_s", "Is_J_per_m2"]].mean().reset_index()
    monthly_melt = monthly.melt(id_vars="MonthLabel", var_name="Variable", value_name="Value")
    name_map = {
        "t2m_C": "ERA5 2m air temp",
        "lblt_C": "ERA5 bottom temp",
        "wind_norm_m_per_s": "Wind speed",
        "Is_J_per_m2": "Solar radiation",
    }
    monthly_melt["Variable"] = monthly_melt["Variable"].map(name_map)
    sns.lineplot(data=monthly_melt, x="MonthLabel", y="Value", hue="Variable", marker="o", ax=axes[0, 0])
    axes[0, 0].set_title("Monthly mean seasonal cycle")
    axes[0, 0].set_xlabel("Month")
    axes[0, 0].tick_params(axis="x", rotation=35)

    sns.boxplot(data=daily, x="MonthLabel", y="wind_norm_m_per_s", color="#80b1d3", ax=axes[0, 1])
    axes[0, 1].set_title("Monthly distribution of wind speed")
    axes[0, 1].set_xlabel("Month")
    axes[0, 1].set_ylabel("Wind speed (m/s)")
    axes[0, 1].tick_params(axis="x", rotation=35)

    season_temp = valid.melt(
        id_vars=["Season"],
        value_vars=["LST_C", "t2m_C", "lblt_C"],
        var_name="Variable",
        value_name="Temperature_C",
    )
    season_temp["Variable"] = season_temp["Variable"].map({
        "LST_C": "MODIS LST",
        "t2m_C": "ERA5 2m air temp",
        "lblt_C": "ERA5 bottom temp",
    })
    sns.boxplot(data=season_temp, x="Season", y="Temperature_C", hue="Variable", ax=axes[1, 0])
    axes[1, 0].set_title("Seasonal temperature distribution on valid LST days")
    axes[1, 0].set_xlabel("Season")
    axes[1, 0].set_ylabel("Temperature (C)")

    lst_count = valid.groupby("MonthLabel", observed=False).size().reindex(MONTH_LABELS, fill_value=0)
    axes[1, 1].bar(lst_count.index, lst_count.values, color="#8da0cb")
    axes[1, 1].set_title("Monthly count of valid MODIS LST observations")
    axes[1, 1].set_xlabel("Month")
    axes[1, 1].set_ylabel("Valid observation days")
    axes[1, 1].tick_params(axis="x", rotation=35)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_matrix(valid, output_path: Path):
    corr_df = valid[["LST_C", "t2m_C", "lblt_C", "wind_norm_m_per_s", "Is_J_per_m2", "lmld_m"]].copy()
    corr_df = corr_df.rename(columns={
        "LST_C": "MODIS LST",
        "t2m_C": "ERA5 2m air temp",
        "lblt_C": "ERA5 bottom temp",
        "wind_norm_m_per_s": "Wind speed",
        "Is_J_per_m2": "Solar radiation",
        "lmld_m": "Mixed layer depth",
    })

    grid = sns.pairplot(
        corr_df,
        corner=True,
        diag_kind="hist",
        plot_kws={"s": 24, "alpha": 0.65, "edgecolor": "none"},
        diag_kws={"bins": 18, "color": "#4c72b0"},
    )
    grid.fig.suptitle("Correlation matrix on valid MODIS LST days", y=1.02)
    grid.fig.savefig(output_path, bbox_inches="tight")
    plt.close(grid.fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize downloaded ERA5 and MODIS LST data.")
    parser.add_argument("--daily", default=None, help="Path to the ERA5 daily CSV.")
    parser.add_argument("--hourly", default=None, help="Path to the ERA5 hourly CSV.")
    parser.add_argument("--lst", default=None, help="Path to the MODIS LST CSV.")
    parser.add_argument("--lake-name", default=None, help="Optional lake name used in figure titles.")
    parser.add_argument("--output-dir", default=None, help="Directory for saved figures.")
    args = parser.parse_args()

    daily_path = resolve_existing_file(args.daily, "ERA5 daily")
    hourly_path = resolve_existing_file(args.hourly, "ERA5 hourly")
    lst_path = resolve_existing_file(args.lst, "MODIS LST")

    daily, hourly, lst, lst_good, lst_other, merged, valid = load_data(daily_path, hourly_path, lst_path)
    metadata = infer_metadata(daily, daily_path, lst_path, args.lake_name)
    default_output_dir = infer_output_dir(daily_path, hourly_path, lst_path, None)
    output_dir = normalize_input_path(args.output_dir).resolve() if args.output_dir else prompt_for_output_dir(default_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = build_output_paths(output_dir, metadata)

    plot_overview(daily, lst_good, lst_other, metadata, outputs["overview"])
    plot_temperature_comparison(merged, outputs["temp_compare"])
    plot_hourly_heatmap(hourly, outputs["hourly_heatmap"])
    plot_monthly_seasonal(daily, valid, outputs["monthly"])
    plot_correlation_matrix(valid, outputs["corr"])

    print("可视化完成:")
    print(outputs["overview"])
    print(outputs["temp_compare"])
    print(outputs["hourly_heatmap"])
    print(outputs["monthly"])
    print(outputs["corr"])
    total_days = int(daily["Date"].dt.normalize().nunique())
    print(f"MODIS 有效 LST 天数: {len(valid)} / {total_days}")


if __name__ == "__main__":
    main()
