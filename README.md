# Mythic M2000 SDK 分析リポジトリ

Mythic M2000 AI アクセラレータ（アナログ compute-in-memory 型）の SDK ツールチェーンを、配布バイナリ・抽出ソースの静的解析によってリバースエンジニアリングした記録。**SDK 本体ではなく、SDK を解析するためのリポジトリ**。

**まずは [doc/reverse-engineering/00_overview.md](doc/reverse-engineering/00_overview.md) から読むこと。**

---

## 構成

このリポジトリは「ベンダー配布物」と「このリポジトリで作成した分析物」を分けている。

| ディレクトリ | 内容 | SDKバージョンが変わったら |
|---|---|---|
| `mythic_sdk/` | **ベンダー配布物一式**。installer zip・展開済みディレクトリ、Docker起動スクリプト(`run_mythic_sdk_container.sh`等)、抽出済みソース(`_extracted_compiler/`・`_extracted_sdk/`)、ベンダー提供ドキュメント(`doc/user-guides/`・`doc/reports/`・`doc/datasheets/`)を配置。**このフォルダを丸ごと新バージョンのもので置き換えることを想定した構成** | フォルダ全体を置き換える |
| `doc/reverse-engineering/` | このリポジトリで作成した分析ドキュメント(コンパイル/PPA推定/精度シミュレーションの解析、HOWTO、計画書等) | そのまま(内容はコード上の記述に基づき更新) |
| `tools/` | 分析作業のために作成した独自スクリプト(CARLA→BEVFormer前処理等)。ベンダーコードではない | そのまま |
| `output/` | 生成物の出力先(動画等)。gitignore対象、再生成可能 | そのまま(空になる) |

### `mythic_sdk/` の内訳

- `mythic_sdk/archive/`, `mythic_sdk/*-installer-*/`, `mythic_sdk/*.zip`, `mythic_sdk/*.tar.gz` — installer配布物の生データ(gitignore対象、巨大)
- `mythic_sdk/run_mythic_sdk_container.sh`, `mythic_sdk/gpu_run_mythic_sdk_container.sh`, `mythic_sdk/load_and_tag_docker_images.sh` — ベンダー提供のDocker起動スクリプト(git管理対象、バージョン非依存)
- `mythic_sdk/_extracted_compiler/`, `mythic_sdk/_extracted_sdk/` — SDKのDockerコンテナから抽出したPythonソース(git管理対象、リバースエンジニアリングの分析対象そのもの)
- `mythic_sdk/doc/` — ベンダー提供のPDFドキュメント(User Guide、Compiler Optimization Report、datasheet等)

### 使用中のSDKバージョン

`mythic_sdk/archive/SDK-VERSION` に記録されている(現在: `v26.05.0`)。起動スクリプトはこのファイルから`VERSION`を自動読み取る(`VERSION`環境変数で上書き可)。

---

## リポジトリのルートパスについて

このリポジトリのディレクトリ自体は特定のSDKバージョンに紐付いた名前(`26.05`)のままだが、**リポジトリ内部のドキュメント・スクリプトはこのルートパスをハードコードしていない**(すべてリポジトリ相対パスで記述)。SDKバージョンが変わっても、`mythic_sdk/`フォルダの中身を入れ替えるだけで分析を継続できる想定。
