# 🌱 Hooxi Carbon AI - 탄소 배출 관리 AI 챗봇

후시파트너스의 고객 상담을 위한 AI 챗봇 시스템입니다. LangGraph 기반의 5개의 노드 + ReAct 아키텍처로 구축되었으며, 탄소 배출량 측정, 배출권 거래, ESG 보고서 작성 등을 지원합니다.

## 📁 프로젝트 구조

```
carbon_AI/
├── react-agent/          # LangGraph 백엔드 (Python)
│   ├── src/
│   │   └── react_agent/
│   │       ├── graph.py           # 5-노드 워크플로우
│   │       ├── tools.py           # 도구 함수들
│   │       ├── mcp_client.py      # MCP 클라이언트
│   │       ├── prompts.py         # 시스템 프롬프트
│   │       ├── state.py           # 상태 정의
│   │       └── configuration.py   # 설정
│   ├── mcp_config.example.json    # MCP 설정 템플릿
│   └── pyproject.toml             # Python 의존성
│
└── agent-chat-ui/        # Next.js 프론트엔드
    ├── src/
    │   ├── components/
    │   │   └── thread/
    │   │       ├── artifact-renderer.tsx  # 아티팩트 렌더러
    │   │       ├── artifact.tsx           # 아티팩트 시스템
    │   │       └── messages/
    │   │           └── ai.tsx             # AI 메시지 렌더링
    │   └── app/
    ├── public/
    │   ├── settings.yaml          # 앱 설정
    │   └── chat-openers.yaml      # 채팅 오프너
    └── package.json               # Node.js 의존성
```

## 🏗️ 아키텍처

### 5-Node ReAct Architecture

```
User Input
    ↓
[1. Classify Node] (Sonnet 4.5)
    → 사용자 의도 분류
    ↓
[2. Route Node] (로직)
    → 필요한 도구 선택
    ↓
[3. Call with Tools] (Sonnet 4.5)
    → 도구 호출 결정
    ↓
[4. Execute Node] (도구 실행)
    → 실제 도구 실행
    ↓
[5. Generate Node] (Haiku 4.5)
    → 응답 생성 + 아티팩트
    ↓
[6. Verify Node] (Sonnet 4.5)
    → 품질 검증 (70점 이상)
    ↓
Response
```

### Intent Categories (고객 여정 기반)

- **SERVICE_INQUIRY**: 서비스 문의
- **EMISSION_ESTIMATE**: 배출량 간단 추정
- **MARKET_INFO**: 배출권 시장 정보
- **REGULATION_INFO**: 규제 정보
- **CONSULTATION_REQUEST**: 상담원 연결

## 🛠️ 주요 기능

### 1. 탄소 배출량 계산
- Scope 1/2/3 배출량 계산
- 업종별 배출 계수 적용
- 인터랙티브 계산기 아티팩트

### 2. 배출권 거래 정보
- KRX 시장 데이터 (실시간)
- AI 기반 매칭 점수
- Mermaid 차트 시각화

### 3. MCP 통합 (Firecrawl)
- 환경부 공식 문서 크롤링
- 최신 규제 정보 검색
- 정부 사이트 스크래핑

### 4. 아티팩트 시스템
- React 컴포넌트 동적 렌더링
- Mermaid 다이어그램
- 사용자가 필요할 때만 표시

## 🚀 시작하기

### 필수 요구사항

- **Node.js**: 18.x 이상
- **Python**: 3.11 이상
- **Poetry** 또는 **pip**
- **LangGraph CLI**: `pip install langgraph-cli`

### 1. 저장소 클론

```bash
git clone https://github.com/your-username/carbon_AI.git
cd carbon_AI
```

### 2. 백엔드 설정 (react-agent)

```bash
cd react-agent

# Poetry 사용 시
poetry install

# pip 사용 시
pip install -e .

# MCP 설정 (선택사항)
cp mcp_config.example.json mcp_config.json
# mcp_config.json에 실제 API 키 입력

# 환경 변수 설정
cp .env.example .env
# .env 파일에 ANTHROPIC_API_KEY 설정
```

### 3. 프론트엔드 설정 (agent-chat-ui)

```bash
cd ../agent-chat-ui

# 의존성 설치
npm install

# 설정 파일 확인
# public/settings.yaml
# public/chat-openers.yaml
```

### 4. 실행

**터미널 1 - 백엔드:**
```bash
cd react-agent
langgraph dev
# → http://localhost:2024
```

**터미널 2 - 프론트엔드:**
```bash
cd agent-chat-ui
npm run dev
# → http://localhost:3000
```

## ⚙️ 환경 변수 설정

### react-agent/.env
```env
ANTHROPIC_API_KEY=your_anthropic_api_key
FIRECRAWL_API_KEY=your_firecrawl_api_key  # MCP 사용 시
TAVILY_API_KEY=your_tavily_api_key         # 웹 검색 시
```

### MCP 설정 (선택사항)

`react-agent/mcp_config.json`:
```json
{
  "mcpServers": {
    "firecrawl": {
      "command": "npx",
      "args": ["-y", "firecrawl-mcp"],
      "env": {
        "FIRECRAWL_API_KEY": "fc-xxxxx"
      }
    }
  }
}
```

## 📊 사용 예시

### 1. 배출량 계산
```
사용자: 우리 회사 배출량 계산해줘. 휘발유 1000L, 전기 5000kWh 사용해.

AI: 계산 결과입니다:
- Scope 1: 2.31 tCO2eq
- Scope 2: 2.39 tCO2eq
- 총계: 4.70 tCO2eq

[계산기 열기] → 오른쪽 패널
```

### 2. 시장 정보
```
사용자: 배출권 시장 현황 보여줘

AI: 현재 KRX 배출권 시장 현황입니다:
- KAU 가격: 14,250원
- 변동률: +2.3%
- 거래량: 125,000톤

[차트 보기] → 오른쪽 패널
```

### 3. 규제 정보 (MCP)
```
사용자: 2024년 배출 계수 규정 알려줘

AI: (Firecrawl이 환경부 사이트 크롤링)
환경부 고시 제2024-234호에 따르면...
```

## 🔧 커스터마이징

### 브랜드 색상 변경
`react-agent/src/react_agent/configuration.py`:
```python
brand_color_primary: str = "#0D9488"  # Teal
```

### 시스템 프롬프트 수정
`react-agent/src/react_agent/prompts.py`:
```python
SYSTEM_PROMPT = """안녕하세요! 후시파트너스의 AI 상담사...
```

### 채팅 오프너 변경
`agent-chat-ui/public/chat-openers.yaml`:
```yaml
chatOpeners:
  - "📊 탄소 배출량 측정 서비스"
  - "💰 배출권을 판매하고 싶어요"
```

## 🛡️ 보안 주의사항

⚠️ **절대 GitHub에 올리면 안 되는 파일:**
- `.env` 파일 (API 키 포함)
- `mcp_config.json` (실제 API 키 포함)
- `node_modules/`
- `__pycache__/`

✅ **대신 사용:**
- `.env.example` (템플릿)
- `mcp_config.example.json` (템플릿)

## 📦 기술 스택

### 백엔드
- **LangGraph** 0.6.10 - 워크플로우 오케스트레이션
- **LangChain** 0.3.27 - LLM 통합
- **Anthropic Claude** - Sonnet 4.5 & Haiku 4.5
- **MCP SDK** 1.3.0 - Model Context Protocol
- **Python** 3.11+

### 프론트엔드
- **Next.js** 15.2.3 - React 프레임워크
- **TypeScript** - 타입 안전성
- **Tailwind CSS** 4 - 스타일링
- **shadcn/ui** - UI 컴포넌트
- **Mermaid** - 다이어그램 렌더링
- **LangGraph SDK** - 스트리밍 & UI

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 라이선스

이 프로젝트는 MIT 라이선스 하에 있습니다.

## 👥 개발자

**Hooxi Partners (후시파트너스)**
- Website: https://hooxipartners.com
- Email: contact@hooxipartners.com

## 🙏 감사의 말

- [LangGraph](https://langchain-ai.github.io/langgraph/) - AI 워크플로우 프레임워크
- [Anthropic](https://anthropic.com) - Claude API
- [Firecrawl](https://firecrawl.dev) - 웹 크롤링 MCP 서버
- [agent-chat-ui](https://github.com/langchain-ai/agent-chat-ui) - UI 템플릿

---

Made with ❤️ by Hooxi Partners
