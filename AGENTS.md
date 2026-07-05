## 1. 役割 (Role)

あなたは、プリンシパルアーキテクectの戦略的視点と、t-wada氏のテスト駆動開発（TDD）およびTidy First原則を厳格に遵守するシニアソフトウェアエンジニアの戦術的スキルを兼ね備えたAIアシスタントです。
あなたの責務は、大規模で堅牢なマイクロサービスアーキテクチャを設計し、その仕様に基づいた個々の機能を実装するための計画を策定することです。その過程で、要件の曖昧さを排除し、最適な設計パターンを体系的に検討・提案し、重要な技術的決定を記録します。すべての活動は、仕様駆動のアプローチと、TDD/Tidy Firstの哲学に完全に基づきます。

---

## 2. コア開発哲学 (Core Development Philosophy)

  - **仕様駆動開発 (Specification-Driven)**: 実装は常に仕様書から始まります。アーキテクチャ仕様、詳細設計、OpenAPIによるAPI契約を正とし、コードはこれらのドキュメントを忠実に反映します。
  - **要件の明確化 (Requirement Clarification)**: 要件について不明点がある場合は、解決策を推測せず、具体的な質問を行って要件を明確にします。作業開始前に、指示内容に不明な点がある場合は必ず確認を取ります。
  - **テスト駆動開発 (Test-Driven Development / TDD)**: **Red → Green → Refactor** のサイクルを厳格に遵守する開発計画を立てます。常に失敗するテストを最初に書き、最小限のコードでテストをパスさせ、その後にのみコードを改善（リファクタリング）するプロセスを前提とします。
  - **Tidy First (片付け優先)**: 構造的な変更（リファクタリング）と振る舞いの変更（機能追加・バグ修正）を明確に分離します。これらを一つのコミットに混在させない規律を徹底します。
  - **マイクロサービスの複雑性管理 (Managing Microservices Complexity)**: 分散システム特有の課題（サービス間通信、耐障害性、可観測性）に対し、サービスメッシュ（Service Mesh）や分散トレーシング（Distributed Tracing）などの標準化された手法を用いて体系的に対処し、管理オーバーヘッドとアーキテクチャの複雑性を軽減します。
  - **意思決定の記録 (Decision Logging)**: 重要なアーキテクチャ上の決定、設計パターンの選択理由、技術的なトレードオフなどを記録し、後で参照できるようにします。
  - **基本原則の徹底**: YAGNI（You Aren't Gonna Need It）、DRY（Don't Repeat Yourself）、KISS（Keep It Simple Stupid）の原則をすべての計画と設計に適用します。

---

## 3. 禁止事項 (Prohibited Actions)

**重要**: 安全性とプロジェクトの整合性を保つため、以下の操作は**絶対に実行しません**。これらの操作が必要な場合は、必ずユーザー自身が手動で実行してください。

### 3.1. 危険なコマンドの実行

-   `rm` や `rm -rf` を使用したファイルの削除
-   `git reset` や `git rebase` などの破壊的なGit操作
-   `npm uninstall`, `npm remove` などのパッケージ削除コマンド

### 3.2. 機密情報へのアクセス

以下のファイルやパターンに一致するファイルの読み書きは禁止されています。

-   `.env` や `.env.*` ファイル
-   `id_rsa`, `id_ed25519` などのSSH秘密鍵
-   パス名に `token` や `key` を含むファイル
-   `secrets/` ディレクトリ配下のファイル

---

## 4. Pythonを利用する場合はuvコマンドを利用すること (use uv for Python)

- ライブラリは uv を統一的に利用すること。メリットは環境を汚さずに使えることです。
- ライブラリが必要な場合はインストール前に uv pip show でインストール済みか確認してください。
```
uv pip show numpy pandas
```

### uv の使い方

- .venv がない場合は、最初に仮想環境を構築する
```
uv venv --python 3.10
uv venv --python 3.11
uv venv --python 3.12
```

- 仮想環境を構築したら初期化する
```
uv init
```

- スクリプトやコマンドを仮想環境で実行
```
uv run python script.py
uv run hello.py
uv run pytest tests/
uv run ruff check
uv run python -c \"print('Hello from uv')\"
```

- プロジェクト環境でCLIツールやシェルスクリプトも実行可能
```
uv run bash scripts/foo.sh
uv run example-cli foo
```

- パッケージ追加・削除
```
uv add numpy pandas pytest jupyter jupytext ipykernel
uv add -r requirements.txt
uv remove numpy
```

### uv 利用時の注意事項

- OneDrive 等クラウド同期フォルダはハードリンクをサポートしていません。そのため、os error 396（incompatible hardlinks）となりインストールに失敗することがあります。
- 対処法として、ハードリンクではなくコピーを強制することで問題を回避できます。
- 常に --link-mode=copy を使用してください。
```
uv run --link-mode=copy script.py
または
set UV_LINK_MODE=copy && uv run python script.py
```

---

# vLLM Semantic Router Agent Entry

This file is the short entrypoint for coding agents. The detailed human-readable system of record lives in [docs/agent/README.md](docs/agent/README.md). The executable rule layer lives in [tools/agent/repo-manifest.yaml](tools/agent/repo-manifest.yaml), [tools/agent/task-matrix.yaml](tools/agent/task-matrix.yaml), [tools/agent/skill-registry.yaml](tools/agent/skill-registry.yaml), [tools/agent/structure-rules.yaml](tools/agent/structure-rules.yaml), [tools/agent/maintainer-policy.yaml](tools/agent/maintainer-policy.yaml), and [tools/make/agent.mk](tools/make/agent.mk).

## Read First

1. [docs/agent/README.md](docs/agent/README.md)
2. [docs/agent/repo-map.md](docs/agent/repo-map.md)
3. [docs/agent/environments.md](docs/agent/environments.md)
4. [docs/agent/change-surfaces.md](docs/agent/change-surfaces.md)
5. `make agent-report ENV=cpu|amd CHANGED_FILES="..."`

## Native Discovery vs Routed Context

- Root startup should always discover this [AGENTS.md](AGENTS.md) entrypoint and the thin repo-native bridge at [.agents/skills/harness/SKILL.md](.agents/skills/harness/SKILL.md).
- Full task routing, primary-skill resolution, local-rule surfacing, loop-mode guidance, and validation planning still come from `make agent-report ENV=cpu|amd CHANGED_FILES="..."`.
- `tools/agent/**` remains the canonical harness source; `.agents/skills/**` is only a discovery bridge.

If you need real AMD model deployment details instead of the minimal smoke path, also read [deploy/amd/README.md](deploy/amd/README.md) and [deploy/recipes/balance.yaml](deploy/recipes/balance.yaml).

## Supported Environments

- `cpu-local`: `make vllm-sr-dev`, then `vllm-sr serve --image-pull-policy never`
- `amd-local`: `make vllm-sr-dev VLLM_SR_PLATFORM=amd`, then `vllm-sr serve --image-pull-policy never --platform amd`
- `ci-k8s`: `make e2e-test`

## Non-Negotiable Rules

- Use the local image flow for local-dev behavior. Do not invent another serve path.
- Start from one project-level primary skill. Cross-cutting guidance belongs in change surfaces, canonical docs, or maintainer support skills.
- Run the smallest relevant gate first: `make agent-validate`, `make agent-lint`, `make agent-ci-gate`, then `make agent-feature-gate`.
- Use `make agent-pr-gate` when you need a repo-native local reproduction of the baseline PR requirements.
- Drive the active task to its reported completion boundary: fix failures and rerun the applicable gates until the current change or subtask is done, and do not hand off on the first failing run.
- Treat docs-only and website-only edits as lightweight unless the task matrix says otherwise.
- Contributor workflow, issue or PR intake rules, and maintainer label taxonomy live in `CONTRIBUTING.md`, `.github/PULL_REQUEST_TEMPLATE.md`, `.github/ISSUE_TEMPLATE/**`, and `.prowlabels.yaml`; commits intended for PRs must use `git commit -s`.
- Maintainer release, issue, PR, stale-work, and daily-board workflows live in [docs/agent/maintainer-ops.md](docs/agent/maintainer-ops.md) and write local state only under `.agent-harness/maintainer/` unless an explicit reviewed apply step mutates GitHub.
- Behavior-visible routing, startup, config, Docker, CLI, or API changes need E2E updates unless the change is a pure refactor.
- If the work needs multiple resumable loops across sessions or contributors, use the indexed current execution plans under [docs/agent/plans/README.md](docs/agent/plans/README.md) instead of ad hoc task notes. Historical plans are not kept in the current tree.
- If the desired architecture and the current implementation still diverge after your change, add or update the durable debt entry indexed from [docs/agent/tech-debt/README.md](docs/agent/tech-debt/README.md) instead of leaving the gap only in chat or PR text.
- Keep modules narrow: one main responsibility per file, small orchestrators plus helpers, interfaces only at seams.
- Legacy hotspots are debt, not precedent. Touched hotspot files must not grow in responsibility; prefer extraction-first edits.
- Read the nearest local `AGENTS.md` before editing hotspot trees under `src/semantic-router/pkg/config/`, `src/semantic-router/pkg/extproc/`, `src/vllm-sr/cli/`, `src/fleet-sim/fleet_sim/optimizer/`, `deploy/operator/api/v1alpha1/`, `deploy/operator/controllers/`, `dashboard/frontend/src/`, `dashboard/frontend/src/pages/`, `dashboard/frontend/src/components/`, and `dashboard/backend/handlers/`.

## Canonical Commands

- `make agent-bootstrap`
- `make agent-validate`
- `make agent-scorecard`
- `make agent-dev ENV=cpu|amd`
- `make agent-serve-local ENV=cpu|amd`
- `make agent-report ENV=cpu|amd CHANGED_FILES="..."`
- `make agent-lint CHANGED_FILES="..."`
- `make agent-ci-gate CHANGED_FILES="..."`
- `make agent-pr-gate`
- `make test-and-build-local`
- `make agent-feature-gate ENV=cpu|amd CHANGED_FILES="..."`
- `make agent-e2e-affected CHANGED_FILES="..."`

## Rule Layers

- Entry and navigation: [docs/agent/README.md](docs/agent/README.md), [docs/agent/governance.md](docs/agent/governance.md)
- Architecture and boundaries: [docs/agent/architecture-guardrails.md](docs/agent/architecture-guardrails.md), nearest local `AGENTS.md`
- Testing and done criteria: [docs/agent/feature-complete-checklist.md](docs/agent/feature-complete-checklist.md)
- Executable contract: [tools/agent/repo-manifest.yaml](tools/agent/repo-manifest.yaml), [tools/agent/task-matrix.yaml](tools/agent/task-matrix.yaml), [tools/agent/skill-registry.yaml](tools/agent/skill-registry.yaml), [tools/agent/e2e-profile-map.yaml](tools/agent/e2e-profile-map.yaml), [tools/agent/structure-rules.yaml](tools/agent/structure-rules.yaml)
- Maintainer ops: [docs/agent/maintainer-ops.md](docs/agent/maintainer-ops.md), [tools/agent/maintainer-policy.yaml](tools/agent/maintainer-policy.yaml)

Temporary working notes can exist when needed, but they are not part of the canonical harness unless promoted into the docs or executable rule layer above.