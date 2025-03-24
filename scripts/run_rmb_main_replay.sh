envs=("HalfCheetah-v3")
src_gravity_variant_pairs=(
  "0.1:gravity-10"
  "0.5:gravity-50"
  "1.0:gravity-100"
  "1.5:gravity-150"
  "3.0:gravity-300"
)
tgt_gravity_variant_pairs=(
  # "0.1:gravity-10"
  # "0.5:gravity-50"
  # "1.0:gravity-100"
  # "1.5:gravity-150"
  "3.0:gravity-300"
)
algorithm=("bt_awac")
info=${1:-""}

# 遍历参数组合
for env in "${envs[@]}"; do
  for pair2 in "${tgt_gravity_variant_pairs[@]}"; do
    for pair1 in "${src_gravity_variant_pairs[@]}"; do
      # 分割 gravity 和 variant
      src_gravity=$(echo $pair1 | cut -d':' -f1)
      src_variant=$(echo $pair1 | cut -d':' -f2)
      tgt_gravity=$(echo $pair2 | cut -d':' -f1)
      tgt_variant=$(echo $pair2 | cut -d':' -f2)

      # 输出当前组合
      echo "Running with env=$env, src_gravity=$src_gravity, tgt_gravity=$tgt_gravity"

      # 运行 Python 程序
      python scripts/rmb_main.py --config scripts/configs/${algorithm}/rpl/rpl.yaml \
        --name $env-$src_variant-$tgt_variant-${algorithm}-rpl-replay$info \
        --env $env\
        --src_gravity $src_gravity \
        --src_variant $src_variant \
        --tgt_gravity $tgt_gravity \
        --tgt_variant $tgt_variant \
        --replay true 
    done
  done
done