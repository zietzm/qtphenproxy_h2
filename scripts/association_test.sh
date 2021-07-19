#!/bin/bash

# Fail if any command fails
set -e


for i in $(seq 1 22);
do
  plink \
    --bgen /data1/deep_storage/ukbiobank/imp_bgen_files/ukb_imp_chr${i}_v3.bgen \
    --sample /data1/deep_storage/ukbiobank/genotypes/ukb41039_imp_chr${i}_v3_s487320.sample \
    --remove /data1/deep_storage/ukbiobank/withdraw41039_20181016.csv \
    --extract /data1/home/mnz2108/git/qtphenproxy_h2/data/All_Stroke_MI_markers.txt \
    --make-bed \
    --out /data1/home/mnz2108/git/qtphenproxy_h2/data/genotypes_chr${i}
done


