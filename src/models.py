from typing import Optional
# src/models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PitchTypeLine(BaseModel):
    pitch_type: str
    n: int
    xslg: Optional[float] = None
    xwoba: Optional[float] = None
    whiff_rate: Optional[float] = None
    swing_rate: Optional[float] = None
    zone_rate: Optional[float] = None
    notes: Optional[str] = None

class HeatmapCell(BaseModel):
    x: int
    y: int
    value: float

class Heatmap(BaseModel):
    pitch_type: str
    grid: List[HeatmapCell]              # normalized grid for plotting

class SprayPoint(BaseModel):
    x: float
    y: float
    outcome: str
    launch_speed: Optional[float] = None
    launch_angle: Optional[float] = None

class HitterReportData(BaseModel):
    player: str
    opponent: str
    date: str
    pitch_lines: List[PitchTypeLine]
    heatmaps: List[Heatmap]
    spray: List[SprayPoint]
    splits: Dict[str, List[PitchTypeLine]] = Field(default_factory=dict)  # {"RHP":[...], "LHP":[...]}
