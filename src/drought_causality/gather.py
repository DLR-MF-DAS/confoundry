import click
import json
import duckdb
from drought_causality.analysis import assemble_timeseries

@click.command()
@click.option('-i', '--input-db', help='Database')
@click.option('-n', '--name-map', help='Name map file')
@click.option('-o', '--output-file', help='Output filename')
@click.option('-t', '--table-name', help='Name of the table to create')
def main(input_db, name_map, output_file, table_name):
    with open(name_map, 'r') as fd:
        name_map = json.load(fd)
    df = assemble_timeseries(input_db, name_map, "ndvi")
    breakpoint()
    conn = duckdb.connect(output_file)
    conn.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
    conn.close()

if __name__ == '__main__':
    main()
