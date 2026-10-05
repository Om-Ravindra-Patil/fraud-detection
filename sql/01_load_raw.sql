-- Load the two Kaggle training files into DuckDB as-is.
-- sample_size = -1 makes DuckDB scan the whole file before choosing column types,
-- so sparse columns are not mistyped from the first few thousand rows.

CREATE OR REPLACE TABLE raw_transaction AS
SELECT *
FROM read_csv('{{ raw_dir }}/train_transaction.csv', header = true, sample_size = -1);

CREATE OR REPLACE TABLE raw_identity AS
SELECT *
FROM read_csv('{{ raw_dir }}/train_identity.csv', header = true, sample_size = -1);
