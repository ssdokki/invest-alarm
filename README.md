# 투자알람 웹앱

크레마지수(매수)와 주간 RSI 매도알람을 **웹 주소 하나로 PC·휴대폰에서 보는 대시보드**,
그리고 **매일 아침 07:30 자동 확인 후 Gmail로 알림**을 보내는 클라우드 작업입니다. PC를 켜 둘 필요가 없습니다.

| 구성 | 역할 | 실행 위치 |
| --- | --- | --- |
| `app.py` | 비밀번호로 들어가는 웹 대시보드. 열 때마다 최신 데이터로 계산 | Streamlit Community Cloud (무료) |
| `job.py` + `.github/workflows/alarm.yml` | 매일 07:30 신호 확인 → 새 신호·주간 현황 메일 | GitHub Actions (무료) |
| `settings.json` | 규칙(비중, 기준점수, 72/50, 18개월)과 메일 주소 | 저장소 |

---

## 배포 순서 (처음 한 번, 약 20분)

### 1단계 — Gmail 앱 비밀번호 만들기
1. Google 계정 → **보안** → **2단계 인증** 을 켭니다. (이미 켜져 있으면 넘어가기)
2. https://myaccount.google.com/apppasswords 접속 → 앱 이름에 `투자알람` 입력 → **만들기**
3. 화면에 나온 **16자리 비밀번호**를 메모합니다. (일반 Gmail 비밀번호는 쓰지 않습니다)

### 2단계 — GitHub 저장소 만들기
1. github.com 로그인(ssdokki) → 오른쪽 위 **+** → **New repository**
2. Repository name: `invest-alarm`, **Private** 선택 → **Create repository**
3. 이 폴더의 파일을 올립니다.
   - 저장소 화면의 **uploading an existing file** 클릭 → 압축을 푼 `invest-alarm` 폴더 **안의 모든 항목**(`.github`, `.streamlit` 폴더 포함)을 끌어다 놓기 → **Commit changes**
   - 올린 뒤 저장소 첫 화면에 `.github` 폴더가 보이는지 꼭 확인하세요.

### 3단계 — 메일 비밀번호 등록
1. 저장소 → **Settings** → 왼쪽 **Secrets and variables** → **Actions**
2. **Secrets** 탭 → **New repository secret**
   - Name: `GMAIL_APP_PASSWORD` / Secret: 1단계의 16자리 → **Add secret**

### 4단계 — 웹 대시보드 배포
1. https://share.streamlit.io 접속 → **Continue with GitHub** 로 로그인 (비공개 저장소 접근 허용)
2. **Create app** → **Deploy a public app from GitHub**
   - Repository: `ssdokki/invest-alarm` / Branch: `main` / Main file path: `app.py`
   - App URL: 원하는 주소 (예: `ssdokki-invest`)
3. **Advanced settings** → **Secrets** 칸에 아래 한 줄 입력 (따옴표 포함)
   ```
   APP_PASSWORD = "원하는비밀번호"
   ```
4. **Deploy** → 1~2분 뒤 비밀번호 화면이 뜨면 완료. 주소를 휴대폰 홈 화면에 추가하면 앱처럼 쓸 수 있습니다.

### 5단계 — 메일에 대시보드 버튼 넣기 + 첫 실행
1. 저장소 → Settings → Secrets and variables → Actions → **Variables** 탭 → **New repository variable**
   - Name: `APP_URL` / Value: 4단계 주소 (예: `https://ssdokki-invest.streamlit.app`)
2. 저장소 → **Actions** 탭 → (처음이면 활성화 버튼 클릭) → **투자알람 메일** → **Run workflow**
3. 1~2분 뒤 초록색 체크가 뜨고 Gmail에 '주간 현황' 메일이 오면 완료입니다.

---

## 평소 사용
- **대시보드**: 주소 접속 → 비밀번호. 오른쪽 위 **새로고침**을 누르면 즉시 최신 값으로 다시 계산합니다.
- **메일**: 매일 07:30 확인. 매수 신호(매월 1일 기준), 매도 알람 켜짐, 매도 신호, 알람 해제, 매주 토요일 주간 현황.
- **규칙 바꾸기**: 저장소의 `settings.json` → 연필 아이콘 → 숫자 수정 → Commit. 앱과 메일 모두 다음 실행부터 반영됩니다.
  미리 결과만 보고 싶으면 대시보드의 **규칙 시뮬레이션** 탭을 쓰세요(저장 안 됨).
- **앱 비밀번호 바꾸기**: share.streamlit.io → 앱 메뉴(⋮) → Settings → Secrets.

## 알아 둘 점
- 무료 Streamlit 앱은 오래 안 열면 잠들어서, 깨우는 데 30초쯤 걸립니다. 메일 알람은 이와 무관하게 매일 동작합니다.
- 주간 봉은 **금요일 종가** 기준입니다(Yahoo 차트는 월요일 날짜로 표시). 장중 값은 '잠정'이고 알람은 확정값으로만 판단합니다.
- CNN 공포탐욕지수는 클라우드 서버 접속을 막을 때가 있어, 받은 값을 `data/fear_greed.csv`에 쌓아 두고 못 받을 때 씁니다.
- GitHub Actions의 07:30 실행은 서버 사정으로 몇 분~수십 분 늦을 수 있습니다.
- 첫 배포 후 확인 권장: 나스닥100(QQQ) 2026-09-08 무렵 주간 RSI ≈ 58.5, 매도 알람 시작 2026년 5월 초(이 앱 표기 5/8).

테스프레소 영상 규칙을 자동 계산하는 도구이며 투자 권유가 아닙니다. 판단과 책임은 본인에게 있습니다.
