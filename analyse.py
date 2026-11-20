import pandas as pd

df = pd.read_csv("sync_matches.csv")

print(df["triplet_spread_ms"].max())