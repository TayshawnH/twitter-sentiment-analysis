import csv
import glob

path = './game_data/*.csv'

lists_from_csv = []
for f in glob.glob(path):
    file = open(f, "r")
    csv_reader = csv.reader(file)
    for row in csv_reader:
        lists_from_csv.append(row[0])

print(lists_from_csv)
