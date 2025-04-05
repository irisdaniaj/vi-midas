import pandas as pd

import csv

input_file = "../data/data_new/filtered_species_tax.csv"
output_file = "../data/data_new/filtered_species_tax_quoted.csv"

with open(input_file, newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile, quoting=csv.QUOTE_ALL)

    for row in reader:
        writer.writerow(row)

print(f'All fields wrapped in quotes and saved to: {output_file}')

