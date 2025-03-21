import pandas as pd

df = pd.read_csv('ordinances.csv')
with open("ordinances.txt", "w", encoding="utf-8") as file:
    for _, row in df.iterrows():
        file.write(f"{row['title']}\n{row['content']}\n\n{'-'*80}\n\n")

print("Data prepared successfully!")
