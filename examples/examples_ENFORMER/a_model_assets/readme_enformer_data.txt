Download `saved_model.bp` at [1], move it to the current directory, and extract via `tar -xf enformer_1.tar.gz`

If reproducing our preprocessing steps, FASTA files for CAGI5 are available to download at [2]. Create a subfolder `hg19/` within the current directory and extract each file therein via `gzip -d chr1.fa.gz`, etc.

See our script `examples/testing/testing_model_ENFORMER.py` for general usage

[1] https://tfhub.dev/deepmind/enformer/1
[2] https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/