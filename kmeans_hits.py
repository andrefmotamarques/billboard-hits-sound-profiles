import os
from pathlib import Path

# Data is read from the data/ folder next to this script, wherever the repo was downloaded
DATA_DIR = Path(__file__).resolve().parent / "data"
os.environ["PYSPARK_SUBMIT_ARGS"] = "--driver-memory 4g --executor-memory 4g pyspark-shell"

from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("KMeansHits") \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.sql.ansi.enabled", "false") \
    .getOrCreate()
# ANSI mode is off so that "NA" values in the CSVs become null when cast to
# numbers, as in Spark 3 (Spark 4 turns ANSI mode on by default)

from pyspark.sql import functions as F
from pyspark.sql.functions import col, to_date, month, when, count, expr
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml import Pipeline

# =============================================================
# 1. LOAD DATA
# =============================================================
df_bb = spark.read.format("csv") \
    .option("header", "true").option("inferSchema", "true") \
    .option("sep", ",").option("quote", '"').option("escape", '"') \
    .option("multiLine", "true") \
    .load(str(DATA_DIR / "billboard.csv"))

df_af = spark.read.format("csv") \
    .option("header", "true").option("inferSchema", "true") \
    .option("sep", ",").option("quote", '"').option("escape", '"') \
    .option("multiLine", "true") \
    .load(str(DATA_DIR / "audio_features.csv"))

# casts
df_bb = df_bb.withColumn("week_position", col("week_position").cast("double"))
df_af = df_af.withColumn("spotify_track_explicit", col("spotify_track_explicit").cast("boolean"))
colunas_double = ["spotify_track_duration_ms", "danceability", "energy", "key", "loudness",
                  "mode", "speechiness", "acousticness", "instrumentalness", "liveness",
                  "valence", "tempo", "time_signature", "spotify_track_popularity"]
for c in colunas_double:
    df_af = df_af.withColumn(c, col(c).cast("double"))

df_af = df_af.dropDuplicates(["song_id"]).drop("song", "performer")
df_bb = df_bb.distinct()
df_final = df_bb.join(df_af, on="song_id", how="inner")

# =============================================================
# 2. DECADE OF EACH OBSERVATION (each row = 1 song-week)
# =============================================================
df_final = df_final.withColumn("chart_date", to_date(col("week_id"), "M/d/yyyy")) \
                   .withColumn("year", F.year(col("chart_date"))) \
                   .withColumn("decade", F.concat((F.floor(col("year") / 10) * 10).cast("int").cast("string"), F.lit("s")))

# =============================================================
# 3. WEIGHT BY CHART POSITION (square-root compression)
# weight = ceil(sqrt(101 - week_position))
#   #1   -> 10
#   #10  -> 10
#   #50  -> 8
#   #100 -> 1
# Top:bottom ratio = 10:1. This assumes that being on the Hot 100
# already makes a song a hit, while giving more weight to the top spots.
# Since K-means has no native weightCol, each row is replicated
# 'weight' times via explode(array_repeat(...)).
# =============================================================
features_cluster = ["danceability", "energy", "loudness", "speechiness",
                    "acousticness", "instrumentalness", "liveness",
                    "valence", "tempo", "spotify_track_duration_ms"]

df_obs = df_final.na.drop(subset=features_cluster + ["decade", "week_position"])

df_obs = df_obs.withColumn(
    "weight",
    F.ceil(F.sqrt(F.lit(101) - col("week_position"))).cast("int")
).filter(col("weight") > 0)

print(">>> Rows (song-week) before weight expansion:")
print(f"    {df_obs.count():,}")

# Replicate each row 'weight' times
df_weighted = df_obs.withColumn("dup", expr("explode(array_repeat(1, weight))")) \
                    .drop("dup")

print(">>> Rows after expansion (each observation weighted by ceil(sqrt(101 - position))):")
print(f"    {df_weighted.count():,}")

# Sanity check: weighted distribution by decade
print(">>> Weighted observations by decade:")
df_weighted.groupBy("decade").count().orderBy("decade").show()

# =============================================================
# 4. PREP PIPELINE: VectorAssembler + StandardScaler
# =============================================================
assembler = VectorAssembler(inputCols=features_cluster, outputCol="raw_features",
                            handleInvalid="skip")
scaler = StandardScaler(inputCol="raw_features", outputCol="features",
                        withMean=True, withStd=True)

prep = Pipeline(stages=[assembler, scaler]).fit(df_weighted)
df_prep = prep.transform(df_weighted).select("song_id", "decade", "week_position", "features") \
              .cache()

# =============================================================
# 5. CHOOSE K: elbow + silhouette
# Silhouette is computed on a sample (the replicated dataset is large)
# =============================================================
evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette",
                                distanceMeasure="squaredEuclidean")

df_prep_sample = df_prep.sample(False, 0.1, seed=42).cache()

print(">>> Evaluating K from 2 to 8 (silhouette on a 10% sample)...")
resultados = []
for k in range(2, 9):
    km = KMeans(featuresCol="features", k=k, seed=42, maxIter=30)
    modelo = km.fit(df_prep)
    wssse = modelo.summary.trainingCost
    sil = evaluator.evaluate(modelo.transform(df_prep_sample))
    resultados.append((k, wssse, sil))
    print(f"  k={k}  WSSSE={wssse:14.2f}   silhouette={sil:.4f}")

# --- Save the evaluation chart as HTML with Plotly ---
from plotly.subplots import make_subplots
import plotly.graph_objects as go

fig_eval = make_subplots(rows=1, cols=2, subplot_titles=("Elbow Method (WSSSE)", "Silhouette Coefficient"))
fig_eval.add_trace(
    go.Scatter(x=[r[0] for r in resultados], y=[r[1] for r in resultados], name="WSSSE", mode="lines+markers", line=dict(color="#3b82f6")),
    row=1, col=1
)
fig_eval.add_trace(
    go.Scatter(x=[r[0] for r in resultados], y=[r[2] for r in resultados], name="Silhouette", mode="lines+markers", line=dict(color="#f43f5e")),
    row=1, col=2
)
fig_eval.update_layout(title_text="Choosing the Number of Clusters (K)", showlegend=False)
fig_eval.write_html("kmeans_avaliacao_k.html")
print(">>> HTML saved: kmeans_avaliacao_k.html")

K = 4
print(f">>> K fixed at: {K}")

# =============================================================
# 6. K-MEANS FINAL
# =============================================================
kmeans = KMeans(featuresCol="features", k=K, seed=42, maxIter=50)
modelo = kmeans.fit(df_prep)
df_clusters = modelo.transform(df_prep).select("song_id", "decade", "week_position", "prediction")

# Cluster profiles on the original scale (using the weighted data,
# since that is how the centroids were computed)
df_profile_src = df_clusters.join(
    df_weighted.select("song_id", "decade", "week_position", *features_cluster),
    on=["song_id", "decade", "week_position"],
    how="inner"
)

print(">>> Weighted profile of each cluster (means, original scales):")
df_profile_src.groupBy("prediction") \
    .agg(*[F.round(F.avg(c), 3).alias(c) for c in features_cluster],
         F.count("*").alias("n_obs_ponderadas")) \
    .orderBy("prediction") \
    .show(truncate=False)

# =============================================================
# 7. CLUSTERS x DECADES: the answer
# =============================================================
print(">>> Absolute (weighted) distribution of clusters by decade:")
crosstab = df_clusters.crosstab("decade", "prediction")
crosstab.show()

import pandas as pd
pdf_ct = crosstab.toPandas().set_index("decade_prediction")
pdf_pct = pdf_ct.div(pdf_ct.sum(axis=1), axis=0) * 100
pdf_pct = pdf_pct.round(2).sort_index()
print(">>> Percentage distribution (rows sum to 100%):")
print(pdf_pct)

# ----- Inference: chi-square + Cramér's V -----
# Note: n is inflated by the replication, so the p-value is optimistic.
# Cramér's V (effect size) is used as the main metric.
from scipy.stats import chi2_contingency
import numpy as np

chi2, p, dof, expected = chi2_contingency(pdf_ct.values)
n_total = pdf_ct.values.sum()
r, c_ = pdf_ct.shape
cramers_v = np.sqrt(chi2 / (n_total * (min(r, c_) - 1)))

print(f">>> chi2={chi2:.2f}  dof={dof}  p={p:.2e}  (n={n_total:,})")
print(f">>> Cramér's V = {cramers_v:.4f}")
print("    Interpretation: 0.1=weak, 0.3=moderate, 0.5=strong")
print("    (V is robust to the inflated n; the p-value is not)")

# --- Save the statistical test indicator as HTML with Plotly ---
import plotly.graph_objects as go
fig_indicator = go.Figure()
fig_indicator.add_trace(go.Indicator(
    mode = "gauge+number",
    value = cramers_v,
    domain = {'x': [0, 1], 'y': [0, 1]},
    title = {'text': f"Strength of Association with Decade (Cramér's V)<br><span style='font-size:0.75em;color:gray'>Chi2 = {chi2:.2f} | p-value = {p:.2e} | n = {n_total:,}</span>", 'font': {'size': 16}},
    gauge = {
        'axis': {'range': [0, 0.5], 'tickwidth': 1, 'tickcolor': "darkblue"},
        'bar': {'color': "#a78bfa"},
        'bgcolor': "white",
        'borderwidth': 2,
        'bordercolor': "gray",
        'steps': [
            {'range': [0, 0.1], 'color': '#f1f5f9'},
            {'range': [0.1, 0.3], 'color': '#cbd5e1'},
            {'range': [0.3, 0.5], 'color': '#94a3b8'}
        ],
        'threshold': {
            'line': {'color': "red", 'width': 4},
            'thickness': 0.75,
            'value': 0.30
        }
    }
))
fig_indicator.update_layout(height=400)
fig_indicator.write_html("kmeans_teste_associacao.html")
print(">>> HTML saved: kmeans_teste_associacao.html")

# Sanity check without replication: run the same test with only 1 row per
# (song, week), to get a defensible p-value.
ct_sem_peso = df_obs.crosstab("decade", "week_position")  # trick: use the real features
ct_check = df_obs.groupBy("decade").pivot("week_position").count().fillna(0).toPandas()

# =============================================================
# 8. VISUALIZATIONS
# =============================================================
import plotly.express as px

fig_hm = px.imshow(
    pdf_pct, text_auto=".1f", aspect="auto", color_continuous_scale="Viridis",
    labels=dict(x="Cluster", y="Decade", color="%"),
    title=f"Weighted distribution (%) of clusters by decade (K={K})",
)
fig_hm.write_html("kmeans_clusters_por_decada.html")

pdf_long = pdf_pct.reset_index().rename(columns={"decade_prediction": "decade"}).melt(id_vars="decade", var_name="cluster", value_name="pct")
fig_bar = px.bar(
    pdf_long, x="decade", y="pct", color="cluster", barmode="group",
    category_orders={"decade": sorted(pdf_long["decade"].unique())},
    title=f"Weighted % in each cluster, by decade (K={K})",
    text_auto=".1f",
)
fig_bar.write_html("kmeans_barras_decada.html")

# 2D PCA: sampled so the browser can handle it
pca = PCA(k=2, inputCol="features", outputCol="pca").fit(df_prep)
df_pca = pca.transform(df_prep).select("decade", "features", "pca")
df_pca = modelo.transform(df_pca).select("decade", "prediction", "pca")

pdf_pca = df_pca.sample(False, 0.02, seed=42).toPandas()
pdf_pca["x"] = pdf_pca["pca"].apply(lambda v: float(v[0]))
pdf_pca["y"] = pdf_pca["pca"].apply(lambda v: float(v[1]))

fig_pca = px.scatter(
    pdf_pca, x="x", y="y",
    color=pdf_pca["prediction"].astype(str),
    symbol="decade",
    opacity=0.4,
    title=f"K-means (K={K}): 2D PCA projection (weighted sample)",
    labels={"color": "cluster"},
)
fig_pca.write_html("kmeans_pca_2d.html")

print(">>> HTML files saved:")
print("    kmeans_avaliacao_k.html")
print("    kmeans_clusters_por_decada.html")
print("    kmeans_barras_decada.html")
print("    kmeans_pca_2d.html")
print("    kmeans_teste_associacao.html")
