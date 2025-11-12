from typing import Optional
from scrape_savant import fetch_pitcher_statcast
from plots_movement import render_movement_plot_png

def build_movement_for_pitcher(
    pitcher_id: int,
    pitcher_name: str,
    start_date: str,
    end_date: str,
    out_dir: str = "build/figures",
    include_density: bool = True,
    normalize_by_throws: bool = False,  # set True if you want arm-side positive HB
) -> Optional[str]:
    df = fetch_pitcher_statcast(pitcher_id, start_date, end_date)
    # guard: some scrapes omit player_name
    if "player_name" not in df.columns:
        df["player_name"] = pitcher_name
    
    # Check if we have the required columns and data before rendering
    required_cols = {"pitch_type", "pfx_x", "pfx_z", "p_throws"}
    if df.empty or not required_cols.issubset(df.columns):
        return None
    
    # Check if we have actual movement data (not just empty dataframe)
    df_with_data = df.dropna(subset=["pitch_type", "pfx_x", "pfx_z"])
    if df_with_data.empty:
        return None
    
    # Need at least 10 pitches to create a meaningful movement visualization
    if len(df_with_data) < 10:
        return None
    
    return render_movement_plot_png(
        df, pitcher_name, out_dir=out_dir,
        include_density=include_density,
        normalize_by_throws=normalize_by_throws
    )
