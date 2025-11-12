# src/normalize.py
from models import HitterReportData, PitchTypeLine, Heatmap, HeatmapCell, SprayPoint

def normalize(raw, player, opponent, date) -> HitterReportData:
    # Map raw fields → our contract. Adjust once I see your 5 scripts.
    lines = [PitchTypeLine(**row) for row in raw["pitch_lines"]]
    heatmaps = []
    for hm in raw.get("heatmaps", []):
        cells = [HeatmapCell(x=c["x"], y=c["y"], value=c["v"]) for c in hm["grid"]]
        heatmaps.append(Heatmap(pitch_type=hm["pitch_type"], grid=cells))
    spray = [SprayPoint(**p) for p in raw.get("spray", [])]
    return HitterReportData(player=player, opponent=opponent, date=date,
                            pitch_lines=lines, heatmaps=heatmaps, spray=spray,
                            splits=raw.get("splits", {}))
