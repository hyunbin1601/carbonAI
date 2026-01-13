# 🚀 CarbonAI 배포 가이드

이 폴더에는 CarbonAI 챗봇을 무료로 배포하는 방법이 정리되어 있습니다.

---

## 📁 파일 목록

### 1. 📘 무료_배포_가이드.md
**가장 상세한 배포 가이드**
- Railway, Vercel, Render 등 다양한 플랫폼 비교
- 단계별 상세 설명
- 문제 해결 방법
- 모니터링 및 유지보수

**읽는데 걸리는 시간**: 10-15분
**대상**: 처음 배포하는 사람, 여러 옵션 비교하고 싶은 사람

---

### 2. ⚡ 빠른배포_스크립트.md
**5분 만에 배포 완료**
- Railway만 사용 (가장 간단)
- 명령어 중심의 빠른 가이드
- 체크리스트 포함

**읽는데 걸리는 시간**: 3-5분
**대상**: 빠르게 배포만 하고 싶은 사람

---

## 🎯 어떤 가이드를 따라야 할까?

### 빠른배포_스크립트.md를 선택하세요 (⚡ 추천)
- ✅ 빠르게 배포만 하고 싶다
- ✅ Railway 사용 예정
- ✅ 복잡한 옵션은 필요 없다

### 무료_배포_가이드.md를 선택하세요
- ✅ 여러 플랫폼 비교하고 싶다
- ✅ 상세한 설명이 필요하다
- ✅ 문제 해결 방법을 미리 알고 싶다
- ✅ 커스텀 도메인 연결 등 추가 기능 필요

---

## 🚀 빠른 시작 (30초)

### 최소 준비물
1. **GitHub 계정** (코드 저장)
2. **Railway 계정** (무료 호스팅)
3. **Anthropic API Key** (AI 모델)

### 배포 순서
1. **백엔드 배포** (2분) → GitHub에 푸시 → Railway 연결
2. **프론트엔드 배포** (2분) → GitHub에 푸시 → Railway 연결
3. **테스트** (30초) → URL 접속 → 채팅 테스트

**총 소요 시간**: 5분

---

## 📊 배포 플랫폼 비교

| 플랫폼 | 무료 기간 | 장점 | 단점 | 추천도 |
|--------|-----------|------|------|--------|
| **Railway** | $5 credit/월 (1-2주) | 🏆 가장 쉬움, 한 곳에서 관리 | Credit 소진 시 중지 | ⭐⭐⭐⭐⭐ |
| **Vercel** | 무제한 | Next.js 최적화, 안정적 | 프론트만 가능 | ⭐⭐⭐⭐ |
| **Render** | 무제한 | Python 지원, 안정적 | 슬립 모드 (30초 대기) | ⭐⭐⭐⭐ |

### 🥇 최고 추천 조합
**Railway (프론트 + 백엔드)**
- 가장 쉽고 빠름
- 한 곳에서 모두 관리
- $5 credit로 1-2주 사용 가능

### 🥈 장기 운영 추천
**Vercel (프론트) + Render (백엔드)**
- 완전 무료 (무제한)
- 프론트엔드는 항상 빠름
- 백엔드만 첫 요청 시 30초 대기

---

## 💰 비용 정리

### 완전 무료 옵션
✅ **Vercel + Render**: 영구 무료
- 프론트: 무제한
- 백엔드: 무제한 (슬립 모드)

### 크레딧 방식
⚠️ **Railway**: 월 $5 credit
- 일반 사용: 1-2주 사용 가능
- Credit 소진 시: 서비스 중지
- 해결: 새 계정 or 유료 플랜

### 유료 플랜
💳 **Railway Pro**: $5/월
💳 **Render Starter**: $7/월

---

## ⚙️ 배포에 필요한 파일들

프로젝트에 이미 포함되어 있습니다:

### 백엔드 (react-agent/)
- ✅ `requirements.txt` - Python 의존성
- ✅ `railway.json` - Railway 설정
- ✅ `Procfile` - Render 설정
- ✅ `.gitignore` - Git 제외 파일
- ✅ `.env.example` - 환경 변수 예시

### 프론트엔드 (agent-chat-ui/)
- ✅ `package.json` - Node.js 의존성
- ✅ `.env.production` - 프로덕션 환경 변수
- ✅ `next.config.js` - Next.js 설정

**모든 파일이 준비되어 있어서 바로 배포 가능!**

---

## 🔧 배포 전 체크리스트

### 필수 확인 사항
- [ ] GitHub 계정 생성
- [ ] Railway/Vercel/Render 계정 생성
- [ ] Anthropic API Key 발급
- [ ] `.env` 파일에 API Key 입력

### 선택 사항
- [ ] LangSmith API Key (모니터링)
- [ ] 커스텀 도메인 (유료)

---

## 📞 도움이 필요하면

### 공식 문서
- Railway: https://docs.railway.app
- Vercel: https://vercel.com/docs
- Render: https://render.com/docs

### 커뮤니티
- Railway Discord: https://discord.gg/railway
- Vercel Discord: https://vercel.com/discord
- Render Community: https://community.render.com

### 이슈 등록
문제가 해결되지 않으면 GitHub 이슈를 등록해주세요.

---

## 🎉 배포 성공 후

### 1. URL 공유
팀원들에게 프론트엔드 URL 공유:
```
https://your-app.railway.app
또는
https://your-app.vercel.app
```

### 2. 모니터링
- Railway/Vercel/Render 대시보드에서 로그 확인
- LangSmith에서 대화 기록 모니터링

### 3. 유지보수
- 코드 수정 후 `git push`만 하면 자동 재배포
- Credit 사용량 주기적 확인

---

**성공적인 배포를 응원합니다!** 🚀

궁금한 점이 있으면 언제든지 물어보세요!
