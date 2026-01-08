import click
import json
from drought_causality.analysis import assemble_timeseries_paths, assemble_timeseries

@click.command()
@click.option('-i', '--input-dir', help='Input directory')
@click.option('-o', '--output-file', help='Output filename prefix')
@click.option('-n', '--names', help='Name json')
def main(input_dir, output_file, names):
    with open(names, 'r') as fd:
        dataset_files = json.load(fd)
    df = assemble_timeseries(input_dir, "ndvi", dataset_files)
    df.to_pickle(output_file + ".pickle")
    df.to_csv(output_file + ".csv")

if __name__ == '__main__':
    main()
