export NEO_DISABLE_MITIGATIONS=1
export RAY_DISABLE_FRACTIONAL_GPUS=1

ray start --head \
  --node-ip-address=localhost \
  --port=6379 \
  --num-cpus=64 \
  --num-gpus=8 \
  --memory=1000000000000 \
  --object-store-memory=500000000000 \
  --object-spilling-directory=/tmp/spillexport