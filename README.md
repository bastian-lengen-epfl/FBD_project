# FinancialBigData

Find the overleaf TeX for the final report here:
https://www.overleaf.com/1673274664kdpgxztdsmdp

What to do? Ideas:

- multiprocessing code splitting it by N chunks (lecture 6, 1:13:40), f.e. send chunks of code of length ... on different cores
- compare dask and multiprocessing. Is there a difference?

- use better write/read formats such as parquet, arrow, hdf5 compared to csv.gz to save the resulting bbo-trade data set (lecture 7, 23:00) and measure times as shown in the lecture there

- write ML algorithm which will trade an asset rlying on BBO-trade book
- possibly add PCA (lecture 9, slide 23/47) and auto-encoders to compress data so our ML algorithm works faster

Tip; extract the AAPL.OQ in a data/raw/TRTH/US/bbo and data/raw/TRTH/equities/US/trade folder
