-- Load the PaySim CSV into DuckDB as-is.
-- PaySim has no transaction ID, so one is added from the file order (which is sorted by step).

CREATE OR REPLACE TABLE raw_paysim AS
SELECT
    row_number() OVER () AS transaction_id,
    *
FROM read_csv('{{ raw_dir }}/{{ raw_file }}', header = true);
