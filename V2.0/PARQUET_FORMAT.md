# LabViz DatasetVersion Parquet Contract v1

Canonical cloud DatasetVersion objects use Parquet schema version `1`.

## Writer profile

- Parquet format version: `2.6`
- data pages: `2.0`
- compression: Zstandard
- dictionary encoding: disabled
- statistics: enabled
- timestamps: UTC, microsecond precision; truncation is rejected
- dataframe index: not stored
- missing values: Arrow nulls, including pandas `NaN`, `NA`, `NaT`, and `None`

## Column types

Input columns are normalized before writing:

| Input meaning | Canonical representation |
| --- | --- |
| Boolean | nullable Arrow boolean |
| Integer | nullable signed 64-bit integer |
| Floating point | nullable 64-bit float |
| Date/time | UTC timestamp at microsecond precision |
| Text, categorical, or mixed object | nullable UTF-8 string |

Column order and exact column names are preserved. Names must be unique. Units are stored beside
each column in the `labviz.schema` metadata document and never inferred from values.

## Integrity and content identity

The rule identifier is `sha256-parquet-bytes-v1`. The content hash is lower-case SHA-256 over the
exact final Parquet byte sequence, including its embedded LabViz schema metadata. StoredObject
and DatasetVersion both record this digest. Every staged object is read back and validated before
the PostgreSQL transaction commits; every reopened object is checked against the recorded hash.

The embedded `labviz.schema` JSON metadata records schema version, hash rule, missing-value rule,
UTC timestamp rule, ordered column names, Arrow types, nullability, and units. Changing any writer
rule requires a new Parquet schema version rather than silently changing v1.

Exact pandas, PyArrow, and Parquet writer package versions are processing provenance rather than
part of the current v1 logical schema. Phase 3 records algorithm and code versions on every
ProcessingRun. Persisting exact package versions in object metadata is tracked as follow-up work;
adding mandatory embedded metadata that changes canonical bytes requires Parquet schema v2.
