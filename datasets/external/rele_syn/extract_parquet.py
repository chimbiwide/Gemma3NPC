import pandas as pd

df = pd.read_parquet("../ReLe.parquet")
df.to_csv("rele.csv")