#!/bin/bash

# Fail if any command fails
set -e

# Filter raw BGEN files for the stroke/MI markers
for i in $(seq 1 22);
do
  # In Plink2 we'd use ref-first as the final flag on --bgen
  plink \
    --bgen /data1/deep_storage/ukbiobank/imp_bgen_files/ukb_imp_chr${i}_v3.bgen \
    --sample /data1/deep_storage/ukbiobank/genotypes/ukb41039_imp_chr${i}_v3_s487320.sample \
    --remove /data1/deep_storage/ukbiobank/withdraw41039_20181016.csv \
    --extract /data1/home/mnz2108/git/qtphenproxy_h2/data/markers/all_markers_plink.txt \
    --biallelic-only strict \
    --make-bed \
    --out /data1/home/mnz2108/git/qtphenproxy_h2/data/genotypes/chr${i}
done

# Gather a list of files that were actually produced (not all chromosomes have one of the variants)
ls -1tr /data1/home/mnz2108/git/qtphenproxy_h2/data/genotypes/*.bed | sed -e 's/\.bed$//' > \
  /data1/home/mnz2108/git/qtphenproxy_h2/scripts/genotypes_files.txt

# Combine the files into a single genotypes file
plink \
  --merge-list /data1/home/mnz2108/git/qtphenproxy_h2/scripts/genotypes_files.txt \
  --make-bed \
  --out /data1/home/mnz2108/git/qtphenproxy_h2/data/genotypes/marker_genotypes

# Convert Plink 1 genotypes file to Plink 2
plink2 \
  --bfile /data1/home/mnz2108/git/qtphenproxy_h2/data/genotypes/marker_genotypes \
  --make-pgen \
  --out /data1/home/mnz2108/git/qtphenproxy_h2/data/genotypes/marker_genotypes

# Convert genotypes to a delimited text file
plink2 \
  --pfile /data1/home/mnz2108/git/qtphenproxy_h2/data/genotypes/marker_genotypes \
  --export A \
  --out /data1/home/mnz2108/git/qtphenproxy_h2/data/genotypes/marker_genotypes
