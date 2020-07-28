import pandas as pd
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import association_rules
df = pd.read_csv (r'C:\Users\arash\Desktop\association-rule-mining\dataframe-python.csv')
df
# pd.set_option("display.max_rows", None, "display.max_columns", None)
print(df)
frequent_itemsets_fp=fpgrowth(df, min_support=0.2, use_colnames=True)
print(frequent_itemsets_fp)
rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.7)
print(rules_fp)
rules_fp