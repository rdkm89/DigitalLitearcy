import requests
import json

# ------- USER VARS -------- #
"""
These should be modified relative to the research question.

See the 'cheatseet' for more information. (Not incuded in Github repo.)

Script created by K.Nielbo.

"""

keywords_src = '~/seeds/seed.txt'
data_dest = '/Users/au564346/Desktop/result.txt'
start_year = 1838
end_year = 1858
familyId = 'berlingsketidende'

# ------- END USER VARS -------- #

base_url = 'http://labs.statsbiblioteket.dk/smurf-services/aviser?q=fulltext:({})familyId:({})py:[{} TO {}]'

def extract(keywords_src, familyId, start_year, end_year):

  keywords = None

  with open(keywords_src, 'r', encoding='utf-8') as key_file:
      keywords = [keyword.strip() for keyword in key_file.readlines()]

  first_pass = True
  rows = []
  i = 0

  for keyword in keywords:

      print("seed {}: {}".format(i, keyword))
      i += 1
      url = base_url.format(keyword, familyId, start_year, end_year)
      response = requests.get(url)
      years = json.loads(response.content.decode('utf-8'))['yearCountsTotal']

      if first_pass:
          rows.extend(create_header(years))

      row = [keyword]
      for year in years:
          row.append(str(year['count']))

      rows.append(row)

      first_pass = False

  return rows


def create_header(years):
  year_row = ['year']
  total_row = ['total']

  for year in years:
      year_row.append(str(year['year']))
      total_row.append(str(year['total']))

  return year_row, total_row


def save_data(rows, data_dest):
  with open(data_dest, 'w', encoding='utf-8') as dest:
      for row in rows:
          dest.write(', '.join(row) + '\n')

if __name__ == '__main__':
  res = extract(keywords_src, familyId, start_year, end_year)
  save_data(res, data_dest)
