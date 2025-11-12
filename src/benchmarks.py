# src/benchmarks.py
pitch_name_map = {
    'FF':'Four-Seam','SI':'Sinker','FT':'Two-Seam','SL':'Slider','ST':'Sweeper',
    'CU':'Curveball','KC':'Knuckle Curve','CH':'Changeup','FS':'Splitter','FC':'Cutter',
    'FO':'Forkball','SC':'Screwball','EP':'Eephus','KN':'Knuckleball','SV':'Slurve','CS':'Slow Curve'
}

benchmarks = {
    'RHP': {
        'FF': {'MaxVelo': 97.0, 'AvgVelo': 95.0, 'IVB': 17.0, 'HB': 7.0, 'Strike %': 65, 'Swing %': 50, 'Whiff %': 22, '% of Strikeouts': 25, '% of Walks': 20, 'ReleaseH': 5.8, 'ReleaseS': 2},
        'FT': {'MaxVelo': 96.0, 'AvgVelo': 94.0, 'IVB': 10.0, 'HB': 14.0, 'Strike %': 64, 'Swing %': 47, 'Whiff %': 16, '% of Strikeouts': 10, '% of Walks': 18, 'ReleaseH': 5.75, 'ReleaseS': 2},
        'SI': {'MaxVelo': 95.0, 'AvgVelo': 93.0, 'IVB': 7.0, 'HB': 15.0, 'Strike %': 63, 'Swing %': 48, 'Whiff %': 14, '% of Strikeouts': 8, '% of Walks': 16, 'ReleaseH': 5.7, 'ReleaseS': 2},
        'FC': {'MaxVelo': 92.0, 'AvgVelo': 90.0, 'IVB': 6.0, 'HB': -3.0, 'Strike %': 64, 'Swing %': 46, 'Whiff %': 18, '% of Strikeouts': 10, '% of Walks': 12, 'ReleaseH': 5.6, 'ReleaseS': 2},
        'SL': {'MaxVelo': 88.0, 'AvgVelo': 86.0, 'IVB': 2.5, 'HB': -6.0, 'Strike %': 62, 'Swing %': 45, 'Whiff %': 28, '% of Strikeouts': 20, '% of Walks': 10, 'ReleaseH': 5.7, 'ReleaseS': 2},
        'ST': {'MaxVelo': 85.0, 'AvgVelo': 83.0, 'IVB': 1.0, 'HB': -15.0, 'Strike %': 60, 'Swing %': 42, 'Whiff %': 30, '% of Strikeouts': 18, '% of Walks': 9, 'ReleaseH': 5.6, 'ReleaseS': 2},
        'CU': {'MaxVelo': 83.0, 'AvgVelo': 80.0, 'IVB': -4.0, 'HB': -6.0, 'Strike %': 60, 'Swing %': 35, 'Whiff %': 27, '% of Strikeouts': 12, '% of Walks': 10, 'ReleaseH': 5.65, 'ReleaseS': 2},
        'KC': {'MaxVelo': 84.0, 'AvgVelo': 81.5, 'IVB': -3.0, 'HB': -5.0, 'Strike %': 59, 'Swing %': 33, 'Whiff %': 25, '% of Strikeouts': 10, '% of Walks': 9, 'ReleaseH': 5.6, 'ReleaseS': 2},
        'CH': {'MaxVelo': 88.0, 'AvgVelo': 86.0, 'IVB': 9.0, 'HB': 14.0, 'Strike %': 64, 'Swing %': 50, 'Whiff %': 32, '% of Strikeouts': 14, '% of Walks': 11, 'ReleaseH': 5.7, 'ReleaseS': 2},
        'FS': {'MaxVelo': 95.0, 'AvgVelo': 93.0, 'IVB': 5.0, 'HB': 10.0, 'Strike %': 63, 'Swing %': 52, 'Whiff %': 25, '% of Strikeouts': 12, '% of Walks': 10, 'ReleaseH': 5.75, 'ReleaseS': 2},
        'FO': {'MaxVelo': 84.0, 'AvgVelo': 82.0, 'IVB': 4.0, 'HB': 9.0, 'Strike %': 62, 'Swing %': 40, 'Whiff %': 20, '% of Strikeouts': 8, '% of Walks': 10, 'ReleaseH': 5.6, 'ReleaseS': 2},
        'SC': {'MaxVelo': 82.0, 'AvgVelo': 80.0, 'IVB': 3.0, 'HB': -8.0, 'Strike %': 58, 'Swing %': 37, 'Whiff %': 21, '% of Strikeouts': 6, '% of Walks': 11, 'ReleaseH': 5.6, 'ReleaseS': 2},
        'SV': {'MaxVelo': 86.0, 'AvgVelo': 83.5, 'IVB': 1.0, 'HB': -8.0, 'Strike %': 61, 'Swing %': 44, 'Whiff %': 27, '% of Strikeouts': 15, '% of Walks': 8, 'ReleaseH': 5.65, 'ReleaseS': 2},
        'CS': {'MaxVelo': 73.0, 'AvgVelo': 70.0, 'IVB': -7.0, 'HB': -4.0, 'Strike %': 58, 'Swing %': 36, 'Whiff %': 24, '% of Strikeouts': 6, '% of Walks': 10, 'ReleaseH': 5.7, 'ReleaseS': 2},
    },
    'LHP': {
        'FF': {'MaxVelo': 96.0, 'AvgVelo': 94.0, 'IVB': 16.0, 'HB': -7.0, 'Strike %': 64, 'Swing %': 48, 'Whiff %': 21, '% of Strikeouts': 24, '% of Walks': 19, 'ReleaseH': 5.7, 'ReleaseS': -2.1},
        'FT': {'MaxVelo': 95.0, 'AvgVelo': 93.0, 'IVB': 9.5, 'HB': -14.0, 'Strike %': 63, 'Swing %': 45, 'Whiff %': 15, '% of Strikeouts': 9, '% of Walks': 17, 'ReleaseH': 5.6, 'ReleaseS': -2.1},
        'SI': {'MaxVelo': 94.5, 'AvgVelo': 92.5, 'IVB': 6.5, 'HB': -15.0, 'Strike %': 62, 'Swing %': 46, 'Whiff %': 13, '% of Strikeouts': 7, '% of Walks': 15, 'ReleaseH': 5.6, 'ReleaseS': -2.1},
        'FC': {'MaxVelo': 91.0, 'AvgVelo': 89.0, 'IVB': 5.5, 'HB': 2.0, 'Strike %': 63, 'Swing %': 44, 'Whiff %': 17, '% of Strikeouts': 9, '% of Walks': 11, 'ReleaseH': 5.6, 'ReleaseS': -2.1},
        'SL': {'MaxVelo': 87.0, 'AvgVelo': 85.0, 'IVB': 2.0, 'HB': 6.0, 'Strike %': 61, 'Swing %': 43, 'Whiff %': 26, '% of Strikeouts': 19, '% of Walks': 9, 'ReleaseH': 5.7, 'ReleaseS': -2.1},
        'ST': {'MaxVelo': 84.0, 'AvgVelo': 82.0, 'IVB': 0.5, 'HB': 14.0, 'Strike %': 60, 'Swing %': 40, 'Whiff %': 29, '% of Strikeouts': 16, '% of Walks': 8, 'ReleaseH': 5.6, 'ReleaseS': -2.1},
        'CU': {'MaxVelo': 82.0, 'AvgVelo': 79.0, 'IVB': -5.0, 'HB': 4.0, 'Strike %': 59, 'Swing %': 34, 'Whiff %': 26, '% of Strikeouts': 11, '% of Walks': 9, 'ReleaseH': 5.6, 'ReleaseS': -2.1},
        'KC': {'MaxVelo': 83.0, 'AvgVelo': 80.5, 'IVB': -4.0, 'HB': 5.0, 'Strike %': 58, 'Swing %': 32, 'Whiff %': 24, '% of Strikeouts': 9, '% of Walks': 8, 'ReleaseH': 5.55, 'ReleaseS': -2.1},
        'CH': {'MaxVelo': 87.0, 'AvgVelo': 85.0, 'IVB': 8.5, 'HB': -14.0, 'Strike %': 63, 'Swing %': 48, 'Whiff %': 31, '% of Strikeouts': 13, '% of Walks': 10, 'ReleaseH': 5.65, 'ReleaseS': -2.1},
        'FS': {'MaxVelo': 94.0, 'AvgVelo': 92.0, 'IVB': 4.0, 'HB': -11.0, 'Strike %': 62, 'Swing %': 50, 'Whiff %': 24, '% of Strikeouts': 11, '% of Walks': 9, 'ReleaseH': 5.7, 'ReleaseS': -2.1},
        'FO': {'MaxVelo': 83.0, 'AvgVelo': 81.0, 'IVB': 3.5, 'HB': -10.0, 'Strike %': 61, 'Swing %': 38, 'Whiff %': 19, '% of Strikeouts': 7, '% of Walks': 10, 'ReleaseH': 5.6, 'ReleaseS': -2.1},
        'SC': {'MaxVelo': 81.0, 'AvgVelo': 79.0, 'IVB': 3.0, 'HB': -9.0, 'Strike %': 56, 'Swing %': 36, 'Whiff %': 20, '% of Strikeouts': 5, '% of Walks': 10, 'ReleaseH': 5.55, 'ReleaseS': -2.1},
        'SV': {'MaxVelo': 85.0, 'AvgVelo': 82.5, 'IVB': 0.5, 'HB': 9.0, 'Strike %': 60, 'Swing %': 43, 'Whiff %': 26, '% of Strikeouts': 14, '% of Walks': 8, 'ReleaseH': 5.6, 'ReleaseS': -2.1},
        'CS': {'MaxVelo': 72.0, 'AvgVelo': 69.0, 'IVB': -7.0, 'HB': 4.0, 'Strike %': 57, 'Swing %': 35, 'Whiff %': 23, '% of Strikeouts': 5, '% of Walks': 10, 'ReleaseH': 5.7, 'ReleaseS': 2.1},
    }
}
