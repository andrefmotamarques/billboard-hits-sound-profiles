# Sound Profiles of Billboard Hot 100 Hits (PySpark K-means)

Unsupervised clustering of 60+ years of Billboard Hot 100 hits (1958 to 2021) by their Spotify audio features, to find out whether hit songs fall into natural "sound profiles" and how those profiles changed across decades.

Team project for the *Big Data Processing and Modelling* course, MSc in Data Science, ISCTE (2025/26), developed with Afonso Claudino.

## Method

- **Data:** Billboard Hot 100 weekly charts joined with Spotify audio features on `song_id`. The unit of analysis is a song-week on the chart (284,136 rows).
- **Weighting:** each row is weighted by chart position, `w = ceil(sqrt(101 - position))`, so top positions count up to 10x more than #100. Spark's K-means has no native weights, so rows are replicated, giving about 2.05 million weighted observations.
- **Features:** 10 standardised audio features (danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration).
- **Model:** Spark MLlib pipeline (`VectorAssembler` and `StandardScaler`) and K-means, with k from 2 to 8 evaluated by elbow (WSSSE) and silhouette on a 10% sample.
- **Time analysis:** cluster by decade contingency table and Cramér's V (used instead of a chi-square p-value, which is meaningless with an inflated n).

## Results

K = 4 was chosen as the smallest k giving distinct, nameable profiles (silhouette 0.34):

| Cluster | Profile | Key traits |
|---|---|---|
| C0 | Pop | high danceability (0.73) and valence (0.73) |
| C1 | High energy | energy 0.75, fast tempo (136 BPM) |
| C2 | Instrumental | instrumentalness 0.70 |
| C3 | Acoustic / orchestral | acousticness 0.57, low energy |

The profiles are clearly associated with decade (Cramér's V = 0.28). Acoustic/orchestral songs made up 71% of hits in the 1950s and about 10% by the 2000s, while pop became dominant from the 1980s (60 to 71%) and high-energy songs grew from the 1990s onwards.

### Reproducibility note

The results above are the ones in the submitted report. Re-running `kmeans_hits.py` from this repository (tested with PySpark 3.5 and 4.2 on a local machine) gives the same four profiles with almost identical centroids, but some of the summary statistics differ:

| Metric | Report | Re-run |
|---|---|---|
| Weighted observations | about 2.05 million | 2,053,341 |
| Silhouette, k = 4 | 0.34 | 0.24 |
| Silhouette, k = 2 | 0.67 | 0.32 |
| Cramér's V (cluster × decade) | 0.28 | 0.26 |
| Acoustic/orchestral share in the 1950s | 71% | 69% |
| Pop share, 1980s to 2000s | 60 to 71% | 46 to 51% |

The main conclusions hold: the same four sound profiles emerge, they are associated with decade, acoustic songs dominate the 1950s and fade afterwards, and pop and high-energy songs take over from the 1980s. The differences are most likely because the report was produced with an earlier iteration of the script in a different Spark environment (the silhouette is computed on a 10% random sample, and K-means initialisation depends on how Spark partitions the data), so exact values can vary between runs on different setups.

## Repository structure

```
kmeans_hits.py        # full PySpark pipeline: load, weight, scale, K-means, evaluation, plots
data/                 # billboard.csv and audio_features.csv (Data.World)
report_en.pdf         # full written report (English)
report_pt.pdf         # original report (Portuguese)
requirements.txt      # Python dependencies
```

## How to run

```bash
pip install -r requirements.txt
python kmeans_hits.py
```

The script reads the data from the `data/` folder next to it, so it can be run from any folder. PySpark needs Java installed (Java 17 or newer for PySpark 4). Interactive Plotly charts are saved as HTML files.

## Tech stack

PySpark (Spark SQL, MLlib) · pandas · NumPy · Plotly

## Team

André Marques and Afonso Claudino. The full report with the team's methodology and discussion is in [`report_en.pdf`](report_en.pdf) (English) and [`report_pt.pdf`](report_pt.pdf) (original, Portuguese).
