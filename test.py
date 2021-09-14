import csv

file = open("data/Apex.csv", "r")
csv_reader = csv.reader(file)

lists_from_csv = []
for row in csv_reader:
    lists_from_csv.append(row)

print(lists_from_csv)
