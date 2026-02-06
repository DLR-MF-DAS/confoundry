import click
import json
from drought_causality.analysis import assemble_timeseries

@click.command()
@click.option('-i', '--input-db', help='Database')
@click.option('-o', '--output-file', help='Output filename prefix')
def main(input_db, output_file):
    df = assemble_timeseries(input_db, "ndvi")
    df.to_pickle(output_file + ".pickle")
    df.to_csv(output_file + ".csv")

if __name__ == '__main__':
    main()
