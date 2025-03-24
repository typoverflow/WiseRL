envs=("HalfCheetah-v3")
gravity_variant_pairs=(
  "0.5:gravity-50"
  "1.0:gravity-100"
  "1.5:gravity-150"
)
algorithm=("oracle_awac")
info=${1:-""}

# 遍历参数组合
for env in "${envs[@]}"; do
  for pair in "${gravity_variant_pairs[@]}"; do
    # 分割 gravity 和 variant
    gravity=$(echo $pair | cut -d':' -f1)
    variant=$(echo $pair | cut -d':' -f2)

    # 输出当前组合
    echo "Running with env=$env, gravity=$gravity, variant=$variant"

    # 运行 Python 程序
    python scripts/main.py --config scripts/configs/${algorithm}/rpl/rpl.yaml \
        --name $env-$variant-${algorithm}-rpl-$info\
        --env $env\
        --gravity $gravity \
        --variant $variant

  done
done