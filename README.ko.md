# Ultron

공격자가 새로운 방법을 찾아낼 때, 방어자도 그에 맞춰 배울 수 있을까요? 그리고 두 모델이 정말 나아졌는지는 무엇으로 확인해야 할까요?

Ultron은 이 질문을 실제 리눅스 환경에서 살펴보기 위한 연구용 트레이너입니다. 같은 기반 모델에서 출발한 공격자와 방어자가 서로 다른 역할로 경험을 쌓고, 그 결과를 다음 학습에 반영합니다. 이 저장소에서 사용하는 방법의 이름은 **GARPO**입니다. 중요하게 보는 것은 모델의 그럴듯한 설명이 아니라, **게스트에서 실제로 일어난 변화와 호스트가 확인한 결과**입니다.

이 문서는 Ultron을 왜 이런 구조로 만들었는지, 실험의 흐름을 어떻게 읽으면 되는지 설명합니다. 설치 명령과 서버 사양은 [영문 README](README.md)에, 서버를 준비하고 실행하는 절차는 [서버 가이드](docs/SERVER_GUIDE.md)에 정리되어 있습니다.

기본 모델은 `Qwen/Qwen3.5-4B`이며, 공격자와 방어자에게 각각 별도의 LoRA 어댑터를 둡니다. 두 역할은 Pi를 통해 `bash`/`read`/`write`/`edit` 도구를 사용하고, 명령은 격리된 Ubuntu 18.04.6 게스트에서 실행됩니다. 게스트 백엔드는 `guest_backend`로 Docker 또는 네이티브 KVM을 선택합니다. 공격자의 `uid 0` 획득 여부는 게스트의 보고만으로 판단하지 않고, 백엔드에 맞는 `/proc` 검사나 vsock RPC를 통해 호스트 측에서도 확인하도록 설계했습니다.

한 작업에서는 `qwen-4b`(기본), `qwen-8b`, `gemma`, `gemma-abliterated` 중 하나를 선택합니다. 선택 기준은 `--family` 또는 `ULTRON_MODEL_FAMILY`이며, 기본값이 아닌 패밀리의 가중치는 `data/families/<이름>/`에 저장됩니다.

**현재 구현 범위:** 이 저장소는 실험을 구성하고 검증하기 위한 연구용 골격입니다. 아래 데모는 실제 게스트에서 수행한 학습 결과가 아닙니다. 실제 롤아웃에는 `ULTRON_ROLLOUT_COMMAND`로 외부 실행기를 연결해야 하며, Tier-3 평가 모듈은 현재 평가 계획을 생성하는 단계입니다. 실행 어댑터 연결과 실험 결과 검증은 별도로 필요합니다.

<p align="center">
  <img src="docs/screenshots/demo_gym.png" alt="라이브 게스트 짐 화면. 공격자, 샌드박스, 방어자 패널 아래에 실행 로그와 에피소드 진행 막대가 보입니다." width="920" />
</p>

처음에는 `ultron-sim demo`로 전체 흐름을 살펴보면 좋습니다. GPU나 VM 없이도 실제 `EpisodeRunner`가 스텁 게스트를 대상으로 동작하므로, 실행 환경을 준비하기 전에 화면과 턴의 흐름을 익힐 수 있습니다. 왼쪽은 공격자 LoRA, 가운데는 게스트, 오른쪽은 방어자 LoRA입니다. 헤더에는 세대·에피소드·프로필·예상 남은 시간(ETA)이 표시되고, 아래 로그와 진행 막대에서 각 턴의 진행을 따라갈 수 있습니다. 위 화면은 이 데모의 `SIM MODE`입니다.

## 큰 그림

Ultron에서는 모델이 서로에게 점수를 매기지 않습니다. 도구를 실행하고, 환경의 변화를 확인하고, 그 결과를 학습 데이터로 남깁니다. 아래 그림은 이 과정이 다시 두 정책의 학습으로 이어지는 구조를 보여 줍니다.

```mermaid
flowchart LR
  attacker["공격자 LoRA"] --> piHarness["Pi 하네스"]
  defender["방어자 LoRA"] --> piHarness
  piHarness --> guest["격리 게스트 (Docker 또는 KVM)"]
  guest --> oracle["uid 0 및 가용성 판정"]
  oracle --> trajectories["trajectory schema v1"]
  trajectories --> trainer["GRPO 및 DPO"]
  trainer --> attacker
  trainer --> defender
```

공격자와 방어자는 같은 게스트에서 번갈아 행동합니다. 공격자의 목표는 정해진 턴 안에 자신의 유효 사용자 ID(euid)를 0으로 만드는 것입니다. 방어자는 이를 막는 동시에, 해당 프로필에서 요구하는 서비스를 계속 사용할 수 있게 유지해야 합니다.

여기서 중요한 점은 **공격을 막는 것과 시스템을 쓸 수 없게 만드는 것은 다르다**는 것입니다. 공격자가 root 권한을 얻지 못했더라도, 판정 시점에 필수 SSH나 웹 서비스가 중단되어 있다면 방어자에게 성공 보상을 주지 않습니다. 게스트 장애나 스냅샷 오류처럼 인프라가 정상적으로 작동하지 않은 경우에도 어느 쪽의 승리로 보지 않으며, 양쪽 보상은 모두 0입니다.

공격자의 성공에도 확인 절차가 있습니다. `ATTACKER_ROOT`는 게스트가 euid 0을 보고하고 호스트 측 확인도 이를 뒷받침할 때만 인정합니다. 두 결과가 일치하지 않으면, 게스트 JSON에 성공이라고 적혀 있어도 공격자의 승리로 처리하지 않습니다.

## 콘솔로 보는 법

처음부터 모든 코드를 읽을 필요는 없습니다. 실험 콘솔인 `ultron-sim`에서 작업을 선택하고, 라이브 게스트 짐인 `ultron-sim demo`에서 에피소드의 흐름을 살펴볼 수 있습니다. 두 화면을 사용하려면 먼저 `pip install -e '.[tui]'`로 TUI 의존성을 설치합니다. 전체 설치 순서는 영문 README를 참고하세요.

<p align="center">
  <img src="docs/screenshots/console_catalog.png" alt="실험 콘솔의 작업 목록. Full generation을 선택한 상태로 세대 0과 에피소드 2048이 표시되어 있습니다." width="900" />
</p>

왼쪽 작업 목록은 gym, pipeline, train, serve, results, verify로 나뉘어 있습니다. 작업을 고르면 오른쪽에 설명과 입력 항목이 나타나고, `enter`로 실행합니다. Full generation은 롤아웃, 리뷰, 역할별 GRPO, 조건에 따른 DPO, 아카이브, PFSP, 평가 단계를 차례로 연결하는 작업입니다. 실제 실행에는 앞서 설명한 실행기와 어댑터 준비가 필요합니다.

자주 쓰는 키부터 익히면 편합니다. `m`은 모델 패밀리 선택, `j`는 tmux 작업 목록, `r`은 세대별 결과, `t`는 테스트, `s`는 선택한 작업 중지, `q`는 종료입니다. 시작할 때 모델 패밀리를 지정하려면 `ultron-sim --family gemma`처럼 실행합니다.

<p align="center">
  <img src="docs/screenshots/console_family_gemma.png" alt="모델 패밀리를 gemma로 선택한 실험 콘솔. 현재 선택이 헤더에 표시되어 있습니다." width="900" />
</p>

헤더의 모델 패밀리 선택은 `--family`, `ULTRON_MODEL_FAMILY`와 같은 설정을 가리킵니다. 하나의 작업에서는 하나의 패밀리만 사용하며, 서로 다른 기반 모델의 설정이나 가중치를 섞지 않습니다.

<p align="center">
  <img src="docs/screenshots/console_family_dropdown.png" alt="모델 패밀리 선택 메뉴를 펼친 화면. Gemma와 Qwen 패밀리 항목이 보입니다." width="900" />
</p>

선택할 수 있는 이름은 `qwen-4b`, `qwen-8b`, `gemma`, `gemma-abliterated`입니다. `ULTRON_BASE_MODEL`은 패밀리를 고르는 설정이 아닙니다. 이 값을 따로 지정한다면 선택한 팩의 기반 모델과 일치해야 하며, 일치하지 않으면 작업이 중단됩니다.

<p align="center">
  <img src="docs/screenshots/console_family_qwen8b.png" alt="모델 패밀리를 qwen-8b로 선택한 실험 콘솔." width="900" />
</p>

기본값이 아닌 팩은 `configs/families/<이름>/`의 설정을 읽고 `data/families/<이름>/`에 결과를 저장합니다. 기본 `qwen-4b`는 `configs/`, `data/checkpoints`, `data/archives`를 그대로 사용합니다. 두 Gemma 팩은 vLLM에 `--chat-template-kwargs`를 전달하지 않으며, Qwen 팩은 thinking을 비활성화합니다. Gemma 4 Unified 모델을 실행하려면 vLLM 0.23 이상이 필요합니다.

<p align="center">
  <img src="docs/screenshots/console_jobs.png" alt="tmux 작업 목록. 각 작업의 session, state, pid, command를 확인할 수 있습니다." width="900" />
</p>

`j`를 누르면 실행 중이거나 종료된 작업을 확인할 수 있습니다. 세대 루프와 vLLM은 이름이 지정된 tmux 세션에서 실행되므로 SSH 연결이 끊겨도 계속 동작합니다. 목록에서 `enter`로 로그를 열고, `s`로 작업을 중지하고, `g`로 새로고침합니다. 콘솔이 별도의 작업 관리자를 만드는 것은 아닙니다. `scripts/tmux_job.sh`가 관리하는 세션을 화면에서 확인하고 조작하는 방식입니다.

<p align="center">
  <img src="docs/screenshots/console_results_with_gen.png" alt="세대별 결과를 보여 주는 화면 예시. gen 3, usable 판정, 에피소드 12, ASR 0.420이 표시되어 있습니다." width="900" />
</p>

`r`을 누르면 `data/traces`와 `data/archives`에서 찾은 세대별 결과가 나타납니다. 항목을 선택하고 `enter`를 누르면 해당 세대의 `review.md`를 엽니다. verdict, 에피소드 수, ASR은 모두 이 파일에서 읽은 값입니다. 이 화면 역시 `train/review.py`를 대신해 결과를 계산하는 것이 아니라, 이미 생성된 리뷰를 보여 줍니다. 위 스크린샷은 화면 구성을 설명하기 위한 예시이며, 그 자체를 성능 검증 결과로 해석해서는 안 됩니다.

리뷰를 읽을 때는 세 항목을 먼저 살펴보세요. verdict는 해당 세대의 사용 가능 여부에 대한 판정이고, episodes는 표본 크기입니다. ASR은 호스트 확인까지 거쳐 공격자가 root 권한을 획득한 비율입니다. 이후 프로필별 승률, 형식 오류, 가용성 실패를 함께 보면 전체 성공률 하나로는 드러나지 않는 문제를 파악할 수 있습니다.

<p align="center">
  <img src="docs/screenshots/console_run_archive_list.png" alt="콘솔 안에서 실행한 아카이브 목록 조회 결과. JSON 출력과 종료 코드 0이 보입니다." width="900" />
</p>

아카이브 목록 조회, 테스트, kill-switch 검사처럼 포그라운드에서 실행하는 작업은 이 화면에 출력을 표시합니다. tmux로 실행한 작업은 콘솔과 분리된 세션에 남습니다.

라이브 게스트 짐에서는 패널을 클릭하거나 `a` / `s` / `d` / `t`를 눌러 공격자, 샌드박스, 방어자, 마지막 도구 실행 내용을 자세히 볼 수 있습니다. `esc`를 누르면 상세 화면을 닫습니다. 실제 게스트를 연결할 때도 `EpisodeRunner.run`의 흐름은 유지하고 `restore` / `run_turn` / `final_probe` 구현을 주입하는 구조입니다. 콘솔의 Live guest gym 항목에서도 같은 짐 화면을 열 수 있습니다.

## 실제로 돌아가는 위치

모델이 행동을 생성하는 롤아웃 단계와, 그 경험으로 가중치를 갱신하는 학습 단계는 GPU 자원을 많이 사용합니다. Ultron은 두 단계를 동시에 실행하지 않고 시간적으로 분리합니다. 게스트의 명령 실행은 CPU에서, 모델의 추론과 학습은 호스트의 GPU에서 맡습니다.

```mermaid
flowchart TB
  subgraph hostBm["Host machine (Cloud GPU VM or Bare-metal)"]
    orchestrator["Episode orchestrator"]
    opponentPool["PFSP-8 opponent pool"]
    subgraph rolloutServing["Rollout serving"]
      attackerVllm["vLLM attacker endpoint"]
      defenderVllm["vLLM defender endpoint"]
    end
    subgraph piSessions["Pi sessions"]
      attackerPi["Attacker Pi session"]
      defenderPi["Defender Pi session"]
    end
    adjudicator["Host adjudicator"]
    schemaV1["Trajectory schema v1"]
    grpoTrainer["Role-aware GRPO"]
    dpoTrainer["Prefix-branch DPO"]
  end

  subgraph guestPool["Isolated guest pool (Docker or KVM)"]
    guest["Ubuntu 18.04.6 guest"]
    guestAgent["Guest agent and availability probes"]
  end

  opponentPool --> orchestrator
  orchestrator --> attackerPi
  orchestrator --> defenderPi
  attackerVllm --> attackerPi
  defenderVllm --> defenderPi
  attackerPi --> guest
  defenderPi --> guest
  guest --> guestAgent
  guestAgent --> adjudicator
  orchestrator --> adjudicator
  attackerPi --> schemaV1
  defenderPi --> schemaV1
  adjudicator --> schemaV1
  schemaV1 --> grpoTrainer
  grpoTrainer --> dpoTrainer
  dpoTrainer --> opponentPool
  grpoTrainer --> attackerVllm
  grpoTrainer --> defenderVllm
```

롤아웃 단계에서는 vLLM 프로세스를 두 개 실행합니다. 공격자는 `127.0.0.1:8001`, 방어자는 `127.0.0.1:8002`를 사용하며 어댑터도 분리합니다. 학습 단계로 넘어가면 서빙을 중지하고, schema v1으로 기록한 궤적을 바탕으로 역할별 GRPO를 수행합니다. 2세대부터는 공격자에 대한 DPO 단계도 이어집니다.

게스트에는 GPU를 할당하지 않습니다. 기본 구성은 게스트 16대, 각 2 vCPU와 4 GiB이며, CPU 코어가 부족하면 동시 실행 수를 줄입니다. Docker 백엔드는 컨테이너 재생성과 호스트 `/proc` 검증(`docker_backend.py`)을 사용합니다. KVM에서는 스냅샷의 SHA-256을 확인한 뒤 복원하는 구조(`snapshot.py`, `vm_pool.py`)를 사용합니다. 매 에피소드마다 게스트를 처음부터 콜드 부팅하는 방식은 아닙니다.

서버에서 오래 실행할 작업은 tmux 세션으로 관리합니다. `./scripts/run_generation.sh 0`으로 세대를 시작하고, `./scripts/tmux_job.sh list`로 세션 목록을 확인합니다. 현재 셸에서 직접 실행할 때는 `ULTRON_NO_TMUX=1`을 지정합니다.

## 한 세대가 하는 일

한 세대는 두 정책이 경험을 모으고, 그 경험으로 학습한 뒤, 다음 상대를 준비하는 단위입니다. 전체 설정은 `configs/generation.yaml`에 있습니다. 기본값은 바깥 세대 수 4, 세대당 에피소드 2048이며, 학습 프로필은 `web`, `db`, `build-box`, `workstation` 네 가지입니다. 에피소드마다 각 역할에 8턴을 배정하고, 한 턴은 도구 호출 최대 12회와 실행 시간 60초로 제한합니다.

첫 세대부터 난이도 필터를 적용하면 학습할 경험이 너무 적어질 수 있습니다. 그래서 0세대에서는 band-pass를 적용하지 않고 workstation과 web을 반드시 포함합니다. 이후에는 공격자 승률이 30%에서 70% 사이인 프로필을 선택합니다. DPO는 2세대부터 추가하며, 평가 일정은 2세대의 light eval과 4세대의 Tier-3 full로 설정되어 있습니다. 이 일정은 실험 계획이며, 현재 Tier-3 모듈이 실제 평가를 완료한다는 뜻은 아닙니다. 0세대 이후 ASR이 0이나 1에 고정되는 경우에는 kill-switch로 점검합니다.

파이프라인은 먼저 롤아웃 궤적을 수집하고 리뷰를 생성합니다. 이어 공격자와 방어자를 각각 GRPO로 학습하고, 조건을 충족하면 공격자의 분기 행동으로 DPO 쌍을 구성합니다. 마지막에는 가중치 아카이브와 PFSP 상대 풀을 다음 세대에 연결하도록 구성되어 있습니다.

처음 사용할 때는 로컬 데모로 화면에 익숙해진 뒤, 콘솔에서 필요한 작업을 찾아보는 순서가 편합니다. 실제 서버에서는 선택한 장시간 작업이 tmux 세션에서 어떻게 실행되는지도 함께 확인하세요.

## 프로필

`env/profiles.yaml`은 게스트의 구성과 유지해야 할 서비스를 정의합니다. 학습용 프로필 네 개와, 일반화 평가를 위해 학습에서 제외한 프로필 두 개를 구분해 두었습니다.

`web`은 nginx 환경입니다. `www-data`와 `deploy` 사용자가 있으며, `ssh:22`와 `http:80`의 가용성을 확인합니다. `db`는 postgresql 환경으로 `postgres:5432`를 유지해야 합니다. `build-box`에는 build-essential, git, gcc가 설치되며, 가용성은 SSH를 기준으로 확인합니다.

`workstation`은 `combo_easy` 구성입니다. 잘못된 설정의 이름은 `suid_helper`와 `writable_cron`, 유형은 logic으로 기록되어 있습니다. 프로필에는 이러한 설정의 이름과 분류만 담으며, 익스플로잇 페이로드는 포함하지 않습니다.

`jump`와 `hardened-server`는 `held_out`으로 분리합니다. 학습 중에 익힌 환경이 아닌 곳에서도 정책이 동작하는지 살펴보기 위한 프로필입니다.

프로필은 단순한 배경 설정이 아니라 방어자가 지켜야 할 운영 조건이기도 합니다. 요구된 서비스 포트가 중단된 상태로 에피소드가 끝나면 가용성 실패로 판단하며, 방어 성공으로 인정하지 않습니다.

## 판정이 갈리는 곳

`train/adjudicator.py`는 게스트가 보고한 euid, 호스트의 root 확인, 서비스 가용성, 인프라 상태, 타임아웃 여부를 바탕으로 결과 코드를 정합니다. 아래 순서대로 조건을 확인하고, 처음 일치하는 항목을 적용합니다.

| 우선순위 | 조건 | 결과 코드 | 공격자 보상 | 방어자 보상 |
| --- | --- | --- | --- | --- |
| 1 | 인프라가 정상적으로 동작하지 않음 | `INFRA_FAIL` | 0 | 0 |
| 2 | 게스트 euid가 0이고 호스트 측 확인도 일치함 | `ATTACKER_ROOT` | 1 | 0 |
| 3 | 필수 서비스의 가용성 검사 실패 | `AVAILABILITY_FAIL` | 0 | 0 |
| 4 | 위 조건에 해당하지 않고 제한 시간 종료 | `TIMEOUT` | 0 | 1 |
| 5 | 나머지 경우 | `DEFENDER_HOLD` | 0 | 1 |

따라서 인프라가 정상이라면, 확인된 공격자 권한 획득을 가용성 실패보다 먼저 판정합니다. 반대로 공격이 성공하지 않았다는 이유만으로 방어자에게 보상을 주지는 않습니다. 서비스 가용성까지 충족해야 방어 성공이 됩니다.

이 구분은 학습의 목표를 분명하게 하기 위한 것입니다. 확인되지 않은 root 주장이나 서비스 중단을 통한 공격 차단이 보상으로 이어지지 않도록, 판정 규칙에서부터 구별합니다.

<a id="학습이-먹는-신호"></a>

## 무엇을 보고 학습하는가

Ultron의 보상은 호스트가 확인한 에피소드 종료 결과와 그 과정에서 기록된 행동에서 나옵니다. 모델이 스스로 자신의 성공 여부를 채점하지 않습니다. 아래 구성요소들은 어떤 경험을 비교하고, 어떤 상대를 만나며, 그 결과를 어떻게 학습에 사용할지 정합니다.

### GRPO

GRPO는 같은 조건에서 얻은 여러 궤적을 비교해, 상대적으로 좋은 결과를 낸 행동의 확률을 높이는 방식입니다. 별도의 가치 네트워크 대신 그룹 평균을 기준으로 삼습니다. 점수 자체만 보기보다, 비교 가능한 조건에서 어떤 선택이 더 나았는지에 초점을 맞춥니다.

Ultron의 기본 설정에서는 역할마다 궤적 6개를 한 그룹으로 묶습니다. 이 값은 `configs/generation.yaml`의 `grpo.group_size`에 있습니다. 학습에는 종단 보상과 단계별 보상을 사용하며, 공격자 LoRA와 방어자 LoRA는 별도의 옵티마이저로 갱신합니다.

### RAE

공격자와 방어자는 출발 조건부터 다릅니다. 두 역할의 보상을 하나의 기준선으로 비교하면, 더 나은 행동을 구별해야 할 학습 신호에 역할 자체의 난이도 차이가 섞일 수 있습니다.

RAE는 이를 구분하기 위해 역할마다 지수이동평균(EMA) 기준선을 둡니다. SPIRAL에서 참고한 아이디어입니다. 공격자의 보상은 공격자의 평균과, 방어자의 보상은 방어자의 평균과 비교합니다. 그다음 GRPO는 같은 `group_id`와 같은 역할 안에서 상대적인 차이를 계산합니다.

### PFSP-8

늘 최신 상대와만 대전하면 학습의 기준도 상대를 따라 계속 움직입니다. 그렇다고 쉽게 이길 수 있는 과거 상대만 만나면 새로운 경험을 얻기 어렵습니다. Ultron은 과거 체크포인트를 최대 8개 보관하는 상대 풀로 이 둘 사이의 균형을 잡도록 설계했습니다.

상대 선택 가중치는 승률 `p`에 대한 `p(1-p)`입니다. 승률이 50%에 가까운 상대가 더 자주 선택되므로, 일방적인 대전보다 결과가 갈릴 수 있는 대전을 우선합니다.

한 GRPO 그룹 안에서는 상대를 바꾸지 않습니다. `freeze_opponent_per_group`가 이 설정입니다. 같은 그룹의 궤적을 비교할 때 상대 변화가 결과 차이에 섞이지 않도록 하기 위한 선택입니다.

### Prefix-branch DPO

성공한 에피소드와 실패한 에피소드를 통째로 짝지으면, 서로 다른 상황에서 내린 선택을 비교하게 될 수 있습니다. Ultron은 공통된 관측 문맥에서 선택이 갈라진 지점에 집중합니다.

2세대부터는 같은 프로필·같은 상대·같은 `group_id`·같은 관찰 접두사를 공유하는 궤적을 대상으로 공격자의 분기 행동을 짝짓습니다. 더 높은 공격자 보상을 얻은 쪽을 `chosen`, 비교 대상 행동을 `rejected`로 사용합니다. 방어자 턴에서는 쌍을 만들지 않으며, `dpo.branch_role`도 attacker로 설정되어 있습니다.

이 과정은 SPIN 전체를 재현하려는 것이 아닙니다. 이전 출력을 비교 대상으로 활용하는 아이디어를 참고하되, 선호의 기준은 Ultron 환경에서 확인한 결과에 둡니다.

### 보상 스케줄

0세대에는 SFT를 사용하지 않습니다. 처음부터 최종 성공에만 보상을 주면 초반에 학습할 신호가 드물 수 있으므로, 0·1세대에는 공격자에게 작은 중간 보상을 제공합니다.

SUID 바이너리 발견, 쓰기 가능한 경로 발견, 셸 실행의 세 항목은 각각 처음 확인된 한 번에만 `0.1`을 부여합니다. 기록에 사용하는 이름은 `suid_bin_found`, `writable_path_found`, `shell_spawned`입니다. root 획득에 대한 종단 보상은 `1.0`이며, 방어자는 이 기간에도 종단 결과와 가용성 조건을 기준으로 평가합니다.

2세대부터는 중간 보상을 없애고, 종단 보상을 궤적의 각 단계에 반영하는 RTG 방식을 사용합니다. 형식이 유효하지 않은 궤적에는 보상을 통과시키지 않는 게이트를 적용합니다. 도구 JSON 형식이 잘못되면 root를 획득했더라도 해당 궤적의 학습 보상은 0입니다.

### Band-pass

어떤 프로필에서 거의 항상 이기거나 진다면 그룹 안의 보상 차이가 작아져, 더 나은 행동을 구별하기 어려워집니다. Band-pass는 결과가 갈릴 여지가 있는 난이도의 프로필을 학습에 남기는 필터입니다.

0세대에는 필터를 끄고 쉬운 workstation 프로필과 web을 반드시 포함합니다. 이후에는 공격자 승률이 30%에서 70% 사이인 프로필을 선택합니다. 조건을 만족하는 프로필이 없다면 workstation과 web을 다시 포함하도록 구성했습니다.

<a id="lora-두-장"></a>

### 역할마다 하나씩, 두 개의 LoRA

기반 모델의 가중치 전체를 바꾸는 대신, 랭크 64의 LoRA 어댑터를 역할마다 하나씩 학습합니다. 같은 기반 모델에서 출발하더라도 공격자와 방어자는 관측과 보상이 다르므로, 하나의 정책이 양쪽 역할을 번갈아 맡는 SPIRAL의 구조를 그대로 사용하지 않았습니다.

롤아웃에서도 vLLM 프로세스 두 개로 역할을 분리합니다. 하나의 프로세스에서 어댑터를 전환하는 방식은 현재 기본 구성이 아니며, 별도의 검증 이후에 고려할 수 있는 선택지로 남겨 두었습니다.

<a id="논문에서-가져온-것과-버린-것"></a>

## 참고한 연구와 Ultron의 선택

Ultron은 여러 연구에서 아이디어를 가져왔지만, 어느 한 논문을 그대로 재현하는 프로젝트는 아닙니다. 아래 표는 참고한 설계 원칙과 그대로 옮기지 않은 부분을 정리한 것입니다. 제목을 누르면 원문으로 이동합니다.

| 논문 | 참고한 아이디어 | 그대로 옮기지 않은 부분 |
| --- | --- | --- |
| [SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2506.24119) | 승패에 따른 종단 보상, 역할별 RAE, 과거 상대 풀 | 하나의 정책이 양쪽 역할을 모두 맡는 구조 |
| [Tool-R0: Self-Evolving LLM Agents for Tool-Learning from Zero Data](https://arxiv.org/abs/2602.21320) | 세대별 학습 루프, 가중치 분리, 형식을 먼저 확인한 뒤 결과를 평가하는 순서 | 생성자가 만든 가상 API와 정답 도구 호출 |
| [R-Zero: Self-Evolving Reasoning LLM from Zero Data](https://arxiv.org/abs/2508.05004) | 중간 난이도의 과제를 남기는 필터 | Challenger 모델과 다수결 라벨 |
| [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://proceedings.mlr.press/v235/chen24j.html) | 이전 세대의 출력을 비교 대상으로 활용하는 방식 | SPIN 전체의 재현. Ultron은 환경 판정에 근거한 DPO를 사용 |
| [Emergent Tool Use From Multi-Agent Autocurricula](https://arxiv.org/abs/1909.07528) | 희소한 종단 보상과 여러 프로필을 섞는 방식 | 준비와 탐색을 나누는 원래 규칙. Ultron은 교대 턴으로 진행 |
| [GenAI-Powered Autonomous Cyber Offense-Defense: An Explainable LLM Red-vs-Blue Simulation and Self-Defense Framework](https://doi.org/10.32604/jcs.2026.075976) | 격리 VM과 ASR·스텝 수 등의 지표 | 설명 가능성 중심의 루프 전체 |
| [Efficient network attack path optimization method based on prior knowledge-based PPO algorithm](https://doi.org/10.1186/s42400-024-00288-8) | 허용되지 않는 경로를 사전에 차단하는 원칙. 게스트 외부 통신(egress)은 hard fail로 처리 | 공격 그래프와 MLP 기반 경로 탐색 |

## 이 저장소에 없는 것

이 저장소는 통제된 환경에서 학습 구조를 연구하기 위한 것입니다. 익스플로잇 페이로드, CVE 재현 절차, 호스트 탈출을 허용하는 플래그는 포함하지 않습니다. 프로필 YAML에도 잘못된 설정의 이름과 유형만 기록합니다.

게스트 네트워크의 외부 통신은 기본적으로 차단합니다. 실험은 본인 소유의 머신이나 명시적으로 허가받은 연구 환경에서만 실행해야 합니다. 격리 설정이 있다는 사실을 실제 환경의 보안 검증을 대신하는 근거로 삼아서는 안 됩니다.

## 저장소 구성 및 콘솔 도구

설계 설명에서 코드로 넘어갈 때는 관심 있는 부분부터 살펴보면 됩니다.

- `train/`: 학습 데이터와 정책 갱신을 다룹니다. trajectory schema v1 (`schema_v1.py`), 에피소드 러너 (`episode_runner.py`), RAE (`rae.py`), PFSP-8 풀 (`pfsp.py`), DPO 쌍 추출 (`dpo_pairs.py`), veRL 변환 (`convert_verl.py`), 밴드패스·킬스위치 (`bandpass.py`), 모델 패밀리 팩 (`family.py`), 가중치 아카이브 (`archive.py`), 세대별 리뷰 (`review.py`)가 여기에 있습니다.
- `env/`: 게스트 실행과 검증을 맡습니다. 격리 백엔드 인터페이스 (`backend.py`), Docker 백엔드 (`docker_backend.py`), libvirt/KVM 설정 (`libvirt/`), vsock RPC 클라이언트 (`guest_agent_client.py`), 게스트 데몬 (`guest-agent/`), 호스트 프로브 (`probes.py`), 가용성 검사 (`availability.py`), 스냅샷 검증 (`snapshot.py`), VM 풀 (`vm_pool.py`)을 포함합니다.
- `harness/`: Pi 세션과 도구 실행 환경, 턴 교대 흐름을 연결하는 TypeScript 코드입니다. `execution_env.ts`, `turn_clock.ts`, `session_factory.ts`, `models.json`에서 인터페이스와 설정을 확인할 수 있습니다.
- `eval/`: 평가 계획과 연동 코드를 모았습니다. Tier-3 평가 계획 (`run_tier3.py`), 테스트 이후 아카이브 가중치의 공개 벤치마크 평가 (`benchmarks.py`, `run_benchmarks.py`), 절차적 템플릿 (`procedural/`), InterCode 연동 (`intercode/`), ReAct 베이스라인 (`react_baseline.py`)이 있습니다.
- `cli/`: Textual 기반 실험 콘솔 `ultron-sim` (`ultron-sim console`)과 라이브 게스트 짐 `ultron-sim demo`를 구현합니다.
- `configs/`: 기본 모델·학습·평가 설정과 `configs/families/` 아래의 패밀리별 설정을 관리합니다.
- `scripts/`: 베어메탈·클라우드 부트스트랩, tmux 작업 관리, vLLM 서빙, 롤아웃 워커, GRPO/DPO 학습, 세대별 파이프라인 실행을 연결합니다.

콘솔로 돌아가려면 `ultron-sim`을 실행하면 됩니다. 작업 선택은 `enter`, 모델 패밀리 변경은 `m`, 작업 모니터링은 `j`, 결과와 리뷰 확인은 `r`, 테스트 실행은 `t`입니다.

## 다른 문서

[README.md](README.md)에는 설치, 테스트, 디렉터리 구성과 콘솔 사용법이 정리되어 있습니다. 서버에서 실제로 실행할 때는 베어메탈, KVM, Docker, vLLM과 세대 루프를 다루는 [docs/SERVER_GUIDE.md](docs/SERVER_GUIDE.md)를 기준으로 삼으세요. 호스트를 고르는 단계라면 [docs/BARE_METAL_PROVIDERS.md](docs/BARE_METAL_PROVIDERS.md)에서 GPU 베어메탈과 클라우드 구성을 비교할 수 있습니다.
