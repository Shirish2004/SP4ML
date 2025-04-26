import pandas as pd

metadata_path = "/home/shirish/Phd/Coursework/sp4ml/utils/data/metadata.csv"

df = pd.read_csv(metadata_path)
class_counts = df["emotion"].value_counts()
print(class_counts)