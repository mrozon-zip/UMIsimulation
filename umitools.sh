samtools view -S -b output.sam > output.bam
samtools sort output.bam -o sorted.bam
umi_tools dedup -I sorted.bam -S deduped.bam --umi-tag=UB --method=directional