source venv/bin/activate
mkdir -p build/pdf
python3 src/generate_report.py --out build/pdf --team NYM --hitter "Pete Alonso" --season_start 2025-03-20 --use-next-series
python3 src/generate_report.py --out build/pdf --team PHI --hitter "Harrison Bader" --season_start 2025-03-20 --use-next-series
python3 src/generate_report.py --out build/pdf --team STL --hitter "Nolan Arenado" --season_start 2025-03-20 --use-next-series
python3 src/generate_report.py --out build/pdf --team PIT --hitter "Tommy Pham" --season_start 2025-03-20 --use-next-series
python3 src/generate_report.py --out build/pdf --team HOU --hitter "Christian Walker" --season_start 2025-03-20 --use-next-series
