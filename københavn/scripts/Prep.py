import pandas as pd

fname = "burialstreets_250418 NU MED FELTNAVNE.xlsx"

streetname_u = list(set(pd.read_excel(fname, header = 0)['streetname']))

df = pd.DataFrame()

df['streetname'] = sorted(streetname_u)

df.to_csv("~/seeds/seed.txt", index = False)
