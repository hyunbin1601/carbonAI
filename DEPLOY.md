# 🚀 Carbon AI 무료 배포 가이드

이 가이드는 Carbon AI 챗봇을 무료로 배포하는 방법을 단계별로 설명합니다.

## 📋 목차
1. [백엔드 배포 (Render)](#1-백엔드-배포-render)
2. [프론트엔드 배포 (Vercel)](#2-프론트엔드-배포-vercel)
3. [환경 변수 설정](#3-환경-변수-설정)
4. [배포 확인](#4-배포-확인)

---

## 1. 백엔드 배포 (Render)

### 1.1 Render 계정 생성
1. [Render](https://render.com) 접속
2. **Sign Up** 클릭 → GitHub 계정으로 로그인

### 1.2 웹 서비스 생성
1. Dashboard에서 **New +** 클릭 → **Web Service** 선택
2. GitHub 리포지토리 연결
   - **Connect a repository** → `hyunbin1601/carbonAI` 검색 및 선택
3. 다음 설정 입력:

```
Name: carbon-ai-backend
Region: Singapore (또는 가까운 지역)
Branch: main
Root Directory: react-agent
Runtime: Python 3
Build Command: pip install -r requirements.txt
Start Command: langgraph serve --host 0.0.0.0 --port $PORT
```

4. **Free** 플랜 선택
5. **Create Web Service** 클릭

### 1.3 환경 변수 설정
서비스 생성 후 **Environment** 탭에서 다음 환경 변수 추가:

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
ANTHROPIC_API_KEY=sk-ant-...
```

> 💡 최소한 OPENAI_API_KEY 또는 ANTHROPIC_API_KEY 중 하나는 필수입니다.

### 1.4 배포 URL 복사
배포가 완료되면 상단에 표시되는 URL을 복사하세요.
예: `https://carbon-ai-backend.onrender.com`

---

## 2. 프론트엔드 배포 (Vercel)

### 2.1 Vercel 계정 생성
1. [Vercel](https://vercel.com) 접속
2. **Sign Up** 클릭 → GitHub 계정으로 로그인

### 2.2 새 프로젝트 생성
1. **Add New** → **Project** 클릭
2. GitHub 리포지토리 선택: `hyunbin1601/carbonAI`
3. **Import** 클릭

### 2.3 프로젝트 설정
```
Framework Preset: Next.js (자동 감지)
Root Directory: agent-chat-ui
Build Command: npm run build
Output Directory: .next
Install Command: npm install
```

### 2.4 환경 변수 설정
**Environment Variables** 섹션에서 다음 변수 추가:

```
NEXT_PUBLIC_API_URL=https://your-backend-url.onrender.com/api
NEXT_PUBLIC_ASSISTANT_ID=agent
LANGGRAPH_API_URL=https://your-backend-url.onrender.com
```

> ⚠️ `your-backend-url.onrender.com`을 실제 백엔드 URL로 변경하세요!

### 2.5 배포
**Deploy** 클릭 → 배포 완료 대기 (약 2-3분)

---

## 3. 환경 변수 설정

### 백엔드 환경 변수 (Render)
```bash
# 필수 (최소 하나)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# 선택 (검색 기능 사용 시)
TAVILY_API_KEY=tvly-...
```

### 프론트엔드 환경 변수 (Vercel)
```bash
# 필수
NEXT_PUBLIC_API_URL=https://carbon-ai-backend.onrender.com/api
NEXT_PUBLIC_ASSISTANT_ID=agent

# 서버 사이드 전용
LANGGRAPH_API_URL=https://carbon-ai-backend.onrender.com
```

---

## 4. 배포 확인

### 백엔드 확인
브라우저에서 백엔드 URL 접속:
```
https://carbon-ai-backend.onrender.com/health
```
정상 작동 시 JSON 응답 확인

### 프론트엔드 확인
Vercel이 제공하는 URL로 접속:
```
https://your-app.vercel.app
```

---

## 🎯 무료 플랜 제한사항

### Render (무료 플랜)
- 15분 비활성 시 자동 슬리핑
- 첫 요청 시 웨이크업 시간 30초 소요
- 750시간/월 무료 실행 시간

### Vercel (무료 플랜)
- 무제한 배포
- 100GB 대역폭/월
- 성능 제한 없음

---

## 🔧 문제 해결

### 백엔드가 슬립 모드로 전환됨
**해결**: 첫 요청 시 30초 정도 대기하면 자동으로 깨어납니다.

### CORS 에러 발생
**해결**: 백엔드 URL이 프론트엔드 환경 변수에 정확히 설정되었는지 확인

### 환경 변수 변경 후 반영 안됨
**해결**:
- Render: **Manual Deploy** 클릭
- Vercel: **Redeploy** 클릭

---

## 📚 추가 옵션

### Railway로 백엔드 배포
Render 대신 Railway를 사용할 수도 있습니다:
1. [Railway](https://railway.app) 가입
2. **New Project** → **Deploy from GitHub repo**
3. 환경 변수 설정
4. 자동 배포

### Netlify로 프론트엔드 배포
Vercel 대신 Netlify를 사용할 수도 있습니다:
1. [Netlify](https://netlify.com) 가입
2. **Add new site** → **Import from Git**
3. 빌드 설정 및 환경 변수 추가

---

## ✅ 완료!

축하합니다! 이제 누구나 접속 가능한 URL로 챗봇이 배포되었습니다.

**배포된 URL:**
- 프론트엔드: `https://your-app.vercel.app`
- 백엔드: `https://carbon-ai-backend.onrender.com`

이 URL을 친구들과 공유하세요! 🎉
