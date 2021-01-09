# FinancialBigData

Find the overleaf TeX for the final report here:
https://www.overleaf.com/1673274664kdpgxztdsmdp

# Writing data to file

### CSV
CSV files are widely used since a majority of table editors are able to create 
and open CSVs and its previews. One can easily display the table using Excel, Numbers etc.
CSV files are row-based. GZIP compression significantly reduces memory usage when storing 
CSV files.

### Parquet
Parquet format uses columnar storage of data, supports very efficient compression 
and encoding schemes. Sticking to this format, one benefits from much faster 
writing and reading times, using less memory when storing files.

### hdf5
The hierarchical data format version 5 is a compressed format. The size of all data 
contained within HDF5 is optimized which makes the overall file size smaller. Hierarchical
technique of breaking the file down to the file system with associated metadata files
allow one to read and write files much faster compared to CSV.

### PyArrow
Apache Arrow is a software development platform for building high performance applications 
that process and transport large data sets. It is not a format in which files can be stored
but this is a packages which allows to process tables faster. It uses in-memory columnar 
format, a standardized, language-agnostic specification for representing structured, 
table-like datasets in-memory. 

### Why do we need them?
Putting it simply, csv.gz is relatively small, but it takes long to write and read; parquet.csv
is super-small and it doesn't take long to write and read, i.e. it outruns csv.gz in every way
but still we can't open it in Excel or Numbers which limits its potential customer base.

### What is done in our project
We write files to three different formats (csv.gz, parquet.gz and hdf5). We also use PA
library to transform a pandas table before writing to parquet. Meantime we are comparing
timings.

# Reading files

### CSV (with and without PyArrow)
Slow. With PyArrow two times faster (wow!).

### Parquet
Very fast, about 6-8 times faster than CSV. Notable that PA slows down reading unlike CSV
case, PyArrow slows down reading by around 1.5 times.

### hdf5
Reading hdf5 file competes with reading a csv file without PyArrow, i.e. relatively slow.
hdf5 is being read slightly faster, but difference is diminishing. 

# Conclusion

We will judge each approach by its reading + writing time and size tradeoff.

- CSV.GZ: average size (~2 Mb), slow. The advantage is that it can be opened in editors
- hdf5: heavy (~13 Mb), but relatively fast.
- Parquet with PA - super great! light and fast. It's better to use this approach.

# PCA and SSA

If we have multiple stocks as an input to our ML algorithm, then it is possible to use
the principle component analysis (PCA) technique to simplify the covariance matrix
leaving out only n first vectors along which the matrix carries most of the information.

Is we have a single stock, it is possible to exploit a single spectrum analysis (SSA)
to decompose the single time-series into its spectrum components and reduce noise and
series size.
