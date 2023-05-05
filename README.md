# DataCollaborationAnalysis
 データ統合解析の数値実験をまとめる

1. 以下コマンドを実行
```
git clone git@github.com:tomos7231/DataCollaborationAnalysis.git
cd DataCollaborationAnalysis
poetry shell
poetry install
```

2. configフォルダ内にyamlファイルを作成
```
# example(config/exp001.yaml)
name: exp001
dataset: movielens # movielens or sushi
seed: 42
num_institution: 2
num_institution_user: 10
num_anchor_data: 1000
dim_intermediate: 500
dim_integrate: 300
```

3. 実験を実行
```
poetry run python main.py exp001
```

以上でoutputフォルダに使用したtrain/testテストと実験結果が保存される。

