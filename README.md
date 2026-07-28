# Mythic M2000 SDK 分析リポジトリ

Mythic M2000 AI アクセラレータ（アナログ compute-in-memory 型）の SDK ツールチェーンを、配布バイナリ・抽出ソースの静的解析によってリバースエンジニアリングした記録。**SDK 本体ではなく、SDK を解析するためのリポジトリ**。

**まずは [doc/reverse-engineering/00_overview.md](doc/reverse-engineering/00_overview.md) から読むこと。**

---

## 構成

このリポジトリは「ベンダー配布物」と「このリポジトリで作成した分析物」を分けている。

| ディレクトリ | 内容 | git | SDKバージョンが変わったら |
|---|---|---|---|
| `mythic_sdk/<version>/` | **ベンダー配布物一式**。バージョン別サブディレクトリ(`v26.05.0/`・`v26.05.2/`…)に、installer zip・展開済みディレクトリ、Docker起動スクリプト(`run_mythic_sdk_container.sh`等)、抽出済みソース(`_extracted_compiler/`・`_extracted_sdk/`)、ベンダー提供ドキュメント(`doc/`)を配置 | **管理外(S3運用)** | 新バージョンのサブディレクトリを追加 |
| `doc/reverse-engineering/` | このリポジトリで作成した分析ドキュメント(コンパイル/PPA推定/精度シミュレーションの解析、HOWTO、計画書等) | 管理対象 | そのまま(内容はコード上の記述に基づき更新) |
| `tools/` | 分析作業のために作成した独自スクリプト(CARLA→BEVFormer前処理等)。ベンダーコードではない | 管理対象 | そのまま |
| `output/` | 生成物の出力先(動画等)。gitignore対象、再生成可能 | 管理外 | そのまま(空になる) |

### `mythic_sdk/` はgit管理外(S3運用)

`mythic_sdk/`配下(SDK実体一式)は**git管理せず、`.gitignore`で丸ごと除外している**。GitHubにはこのリポジトリで作成した分析物(`doc/`・`tools/`・`README.md`等)のみを上げ、SDK配布物・抽出ソースはアップロードしない。SDK実体は必要に応じてS3から取得する:

```
s3://mythic-sdk/<version>/     # 例: v26.05.0/, 26.05.2/, v25.11.0 SDK/
```

各バージョンのサブディレクトリ(`mythic_sdk/v26.05.0/`等)の内訳:

- `archive/`, `*-installer-*/`, `*.zip`, `*.tar.gz` — installer配布物の生データ(S3から取得・展開)
- `run_mythic_sdk_container.sh`, `gpu_run_mythic_sdk_container.sh`, `load_and_tag_docker_images.sh` — ベンダー提供のDocker起動スクリプト
- `_extracted_compiler/`, `_extracted_sdk/` — SDKのDockerコンテナから抽出したPythonソース(リバースエンジニアリングの分析対象そのもの)。`doc/reverse-engineering/`はこれらをリポジトリ相対パスで引用する
- `doc/` — ベンダー提供のPDFドキュメント(User Guide、Compiler Optimization Report、datasheet等)

### 使用中のSDKバージョン

各バージョンサブディレクトリの`archive/SDK-VERSION`に記録されている(例: `mythic_sdk/v26.05.0/archive/SDK-VERSION` → `v26.05.0`)。起動スクリプトは自分と同階層のこのファイルから`VERSION`を自動読み取る(`VERSION`環境変数で上書き可)。

---

## リポジトリのルートパスについて

このリポジトリのディレクトリ自体は特定のSDKバージョンに紐付いた名前(`26.05`)のままだが、**リポジトリ内部のドキュメント・スクリプトはこのルートパスをハードコードしていない**(すべてリポジトリ相対パスで記述)。SDKバージョンが変わっても、`mythic_sdk/`配下に新しいバージョンサブディレクトリを追加するだけで分析を継続できる想定。
