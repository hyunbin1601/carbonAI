"""Custom tools for Hooxi Partners carbon emission management.

This module provides specialized tools for:
1. SearchTool - RAG search through ChromaDB
2. CalculatorTool - Scope 1/2/3 emission calculations (내부 계산)
3. MarketTool - KRX emission trading market data (MCP)
4. ConsultationTool - Naver Works integration (MCP)
5. MCP Tools - Firecrawl and KRX API integration
6. Skills Tools - Document generation (DOCX, XLSX, PDF)
7. Artifact Tools - Interactive visualizations
"""

import json
import time
from typing import Any, Callable, Dict, List, Optional, cast

import aiohttp
from langchain_tavily import TavilySearch

from react_agent.configuration import Configuration


# ============================================================================
# 1. SearchTool - RAG Search through ChromaDB
# ============================================================================


async def search_emission_database(query: str) -> Dict[str, Any]:
    """Search emission factors and regulations from ChromaDB vector store.

    This function performs RAG (Retrieval Augmented Generation) search
    through 100+ documents about emission factors, regulations, and guidelines.

    Args:
        query: Search query (e.g., "2024 전력 배출 계수", "Scope 3 물류 계산")

    Returns:
        {
            "query": "2024 전력 배출 계수",
            "results": [
                {
                    "content": "2024년 전력 배출 계수는 0.4781 kgCO2eq/kWh입니다...",
                    "source": "환경부 고시 제2024-234호",
                    "relevance_score": 0.95
                }
            ],
            "total_results": 5
        }
    """
    # TODO: Implement ChromaDB integration
    # For now, return service information as mock data

    # Service information templates for common queries
    service_info = {
        "측정": {
            "content": """📊 **탄소 배출량 측정 서비스**

▪️ **서비스 내용**: Scope 1/2/3 전체 배출량 정밀 측정, 배출원별 상세 분석, 공식 인증서 발급
▪️ **진행 기간**: 약 2-3주 (현장 조사 포함)
▪️ **제공 자료**: 배출량 보고서, 배출원 분석표, 감축 권고사항, 정부 제출용 인증서
▪️ **적용 대상**: 연 배출량 25,000 tCO2eq 이상 기업 (의무 대상), 자발적 측정 희망 기업

💡 **다음 단계**: 전문 컨설턴트와 상담 → 현장 조사 → 데이터 수집 → 배출량 계산 → 보고서 작성""",
            "source": "후시파트너스 서비스 가이드",
            "relevance_score": 0.98
        },
        "거래": {
            "content": """💰 **배출권 거래 중개 서비스**

▪️ **서비스 내용**: KRX 배출권 시장 매매 중개, AI 기반 최적 거래 매칭, 거래 전략 컨설팅
▪️ **거래 가능**: KAU(한국 배출권), KOC(상쇄 배출권), KCU(외부사업 배출권)
▪️ **수수료**: 거래 성사 시 매매 금액의 0.5% (업계 최저 수준)
▪️ **추가 서비스**: 실시간 시장 정보 제공, 거래 시점 자동 알림, 포트폴리오 관리

💡 **다음 단계**: 보유 배출권 현황 확인 → 거래 전략 수립 → 매칭 진행 → 계약 체결""",
            "source": "후시파트너스 서비스 가이드",
            "relevance_score": 0.98
        },
        "보고서": {
            "content": """📄 **ESG 보고서 작성 대행 서비스**

▪️ **서비스 내용**: 전문 컨설턴트의 맞춤형 ESG 보고서 작성, 정부 제출용 공식 양식 작성, 투자자용 리포트
▪️ **보고서 종류**: 온실가스 배출량 명세서, 지속가능경영 보고서, ESG 경영 현황 리포트
▪️ **진행 기간**: 약 1-2주 (자료 제공 기준)
▪️ **제공 형식**: DOCX, XLSX, PDF (선택 가능)

💡 **다음 단계**: 필요 자료 안내 → 자료 수집 → 보고서 초안 작성 → 검토 및 수정 → 최종본 전달""",
            "source": "후시파트너스 서비스 가이드",
            "relevance_score": 0.98
        },
        "컨설팅": {
            "content": """🎯 **탄소 감축 컨설팅 서비스**

▪️ **서비스 내용**: 배출원 상세 분석, 감축 방안 제시, 정부 지원 사업 연계, 장기 로드맵 수립
▪️ **진행 기간**: 약 4-6주
▪️ **산출물**: 감축 전략 보고서, 실행 로드맵, ROI 분석, 정부 지원금 신청 지원
▪️ **추가 혜택**: 감축 설비 도입 시 협력사 연계

💡 **다음 단계**: 현황 진단 → 감축 목표 설정 → 전략 수립 → 실행 계획 작성""",
            "source": "후시파트너스 서비스 가이드",
            "relevance_score": 0.98
        }
    }

    # Match query to service category
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in ["측정", "배출량", "계산", "measurement"]):
        result = service_info["측정"]
    elif any(keyword in query_lower for keyword in ["거래", "배출권", "krx", "trading"]):
        result = service_info["거래"]
    elif any(keyword in query_lower for keyword in ["보고서", "리포트", "report"]):
        result = service_info["보고서"]
    elif any(keyword in query_lower for keyword in ["컨설팅", "감축", "상담", "consulting"]):
        result = service_info["컨설팅"]
    else:
        # Default: general service overview
        result = {
            "content": f"""후시파트너스는 탄소 배출 관리 전문 기업입니다.

📋 **주요 서비스**:
1. 탄소 배출량 측정 서비스
2. 배출권 거래 중개 서비스
3. ESG 보고서 작성 대행
4. 탄소 감축 컨설팅

💬 더 자세한 정보를 원하시면 구체적인 서비스명을 말씀해주시거나, 전문 상담원 연결을 요청해주세요.

검색어: {query}""",
            "source": "후시파트너스 서비스 가이드",
            "relevance_score": 0.75
        }

    return {
        "query": query,
        "results": [result],
        "total_results": 1
    }


# ============================================================================
# 2. CalculatorTool - Scope 1/2/3 Emission Calculations (내부 계산)
# ============================================================================


async def calculate_emissions(
    fuel_liters: float = 0,
    electricity_kwh: float = 0,
    logistics_km: float = 0,
    natural_gas_m3: float = 0,
) -> Dict[str, Any]:
    """Calculate carbon emissions across Scope 1, 2, and 3.

    배출량 계산은 내부 로직으로 처리합니다 (외부 API 불필요).
    계산식: 배출량 = 활동량 × 배출 계수

    Emission Factors (2024):
    - Gasoline: 2.31 kgCO2/L (Scope 1)
    - Diesel: 2.62 kgCO2/L (Scope 1)
    - Natural Gas: 2.23 kgCO2/m³ (Scope 1)
    - Electricity: 0.4781 kgCO2/kWh (Scope 2)
    - Logistics: 0.185 kgCO2/km (Scope 3)

    Args:
        fuel_liters: Gasoline consumption in liters
        electricity_kwh: Electricity consumption in kWh
        logistics_km: Logistics distance in kilometers
        natural_gas_m3: Natural gas consumption in m³

    Returns:
        {
            "scope1": 456.78,  # tCO2eq
            "scope2": 345.67,  # tCO2eq
            "scope3": 432.11,  # tCO2eq
            "total": 1234.56,  # tCO2eq
            "breakdown": {
                "fuel": 231.0,
                "natural_gas": 225.78,
                "electricity": 345.67,
                "logistics": 432.11
            },
            "factors_used": {
                "fuel": 2.31,
                "electricity": 0.4781,
                "logistics": 0.185,
                "natural_gas": 2.23
            }
        }
    """
    # Emission factors (kgCO2eq per unit)
    FACTORS = {
        "fuel": 2.31,  # kgCO2/L (gasoline)
        "electricity": 0.4781,  # kgCO2/kWh
        "logistics": 0.185,  # kgCO2/km
        "natural_gas": 2.23,  # kgCO2/m³
    }

    # Calculate emissions in kgCO2eq
    fuel_emissions = fuel_liters * FACTORS["fuel"]
    natural_gas_emissions = natural_gas_m3 * FACTORS["natural_gas"]
    electricity_emissions = electricity_kwh * FACTORS["electricity"]
    logistics_emissions = logistics_km * FACTORS["logistics"]

    # Scope categorization
    scope1 = (fuel_emissions + natural_gas_emissions) / 1000  # Convert to tCO2eq
    scope2 = electricity_emissions / 1000
    scope3 = logistics_emissions / 1000
    total = scope1 + scope2 + scope3

    return {
        "scope1": round(scope1, 2),
        "scope2": round(scope2, 2),
        "scope3": round(scope3, 2),
        "total": round(total, 2),
        "breakdown": {
            "fuel": round(fuel_emissions / 1000, 2),
            "natural_gas": round(natural_gas_emissions / 1000, 2),
            "electricity": round(electricity_emissions / 1000, 2),
            "logistics": round(logistics_emissions / 1000, 2),
        },
        "factors_used": FACTORS,
        "disclaimer": "⚠️ 이 계산은 간단한 추정치입니다. 정확한 배출량 측정 및 공식 인증서 발급은 후시파트너스 전문 컨설턴트에게 문의해주세요. (상담 연결: '상담원 연결해주세요'라고 말씀해주세요)",
        "service_recommendation": "정확한 배출량 측정을 원하시면 '탄소 배출량 측정 서비스'를 이용해주세요.",
    }


# ============================================================================
# 3. MarketTool - KRX Emission Trading Market (MCP)
# ============================================================================


async def get_krx_market_data(credit_type: str = "KAU") -> Dict[str, Any]:
    """Get real-time KRX emission trading market data via MCP.

    Args:
        credit_type: Type of emission credit ("KAU", "KOC", "KCU")

    Returns:
        {
            "type": "KAU",
            "price": 14250,  # KRW
            "change": 2.3,  # %
            "volume": 125000,  # tons
            "high": 14500,
            "low": 14000,
            "timestamp": "2025-12-29T10:30:00"
        }
    """
    # TODO: Implement actual KRX API integration via MCP
    # For now, return mock data
    return {
        "type": credit_type,
        "price": 14250,
        "change": 2.3,
        "volume": 125000,
        "high": 14500,
        "low": 14000,
        "timestamp": "2025-12-29T10:30:00",
        "status": "mock_data - KRX MCP 연동 예정"
    }


async def ai_matching_score(
    company_emissions: float,
    target_price: float,
    target_volume: float
) -> Dict[str, Any]:
    """Calculate AI-based matching score for emission trading.

    Args:
        company_emissions: Company's total emissions in tCO2eq
        target_price: Target price per ton (KRW)
        target_volume: Target trading volume in tons

    Returns:
        {
            "match_score": 85,  # 0-100
            "recommended_trades": [
                {
                    "counterparty": "Company A",
                    "volume": 1000,
                    "price": 14200,
                    "score": 92
                }
            ],
            "market_analysis": "현재 시장 상황이 매수에 유리합니다..."
        }
    """
    # Simplified AI matching logic (내부 계산)
    match_score = min(100, int((target_volume / max(company_emissions, 1)) * 100))

    return {
        "match_score": match_score,
        "recommended_trades": [
            {
                "counterparty": "Mock Company A",
                "volume": min(1000, target_volume),
                "price": target_price,
                "score": match_score
            }
        ],
        "market_analysis": f"배출량 {company_emissions}tCO2eq 기준, 목표 거래량 {target_volume}톤은 {'적절' if match_score > 70 else '검토 필요'}합니다."
    }


# ============================================================================
# 4. ConsultationTool - Naver Works Integration (MCP)
# ============================================================================


async def request_human_consultation(
    topic: str,
    urgency: str = "normal"
) -> Dict[str, Any]:
    """Request human consultant connection via Naver Works MCP.

    Args:
        topic: Consultation topic (e.g., "배출권 거래 전략", "감축 계획 수립")
        urgency: Urgency level ("low", "normal", "high")

    Returns:
        {
            "status": "requested",
            "ticket_id": "CONSULT-20251229-001",
            "estimated_response_time": "30 minutes",
            "assigned_consultant": "김철수 전문위원"
        }
    """
    ticket_id = f"CONSULT-{time.strftime('%Y%m%d')}-{int(time.time()) % 1000:03d}"

    response_times = {
        "low": "2 hours",
        "normal": "30 minutes",
        "high": "10 minutes"
    }

    return {
        "status": "requested",
        "ticket_id": ticket_id,
        "estimated_response_time": response_times.get(urgency, "30 minutes"),
        "assigned_consultant": "후시파트너스 상담원",
        "topic": topic,
        "naver_works_notification": "MCP 연동 예정"
    }


# ============================================================================
# 5. MCP Tools - Firecrawl & KRX API
# ============================================================================


async def firecrawl_government_docs(query: str) -> Dict[str, Any]:
    """Crawl government websites (환경부) for latest regulations and emission factors.

    Uses Firecrawl MCP to search and scrape official documents.

    Args:
        query: Search query (e.g., "2024 배출 계수", "탄소중립 규제")

    Returns:
        {
            "documents": [
                {
                    "title": "환경부 고시 제2024-234호",
                    "url": "https://me.go.kr/...",
                    "content": "전력 배출 계수: 0.4781...",
                    "published_date": "2024-11-15",
                    "relevance": 0.95
                }
            ],
            "total_found": 5,
            "search_query": query
        }
    """
    try:
        # Import MCP client
        from react_agent.mcp_client import search_web, scrape_url

        # Search for government documents related to query
        search_query = f"{query} site:me.go.kr OR site:korea.kr"
        search_results = await search_web(search_query, limit=5)

        if search_results.get("success"):
            documents = []
            results = search_results.get("results", [])

            # Scrape each result to get full content
            for idx, result in enumerate(results[:3]):  # Limit to top 3 for performance
                url = result.get("url", "")
                title = result.get("title", f"Document {idx + 1}")

                # Scrape the page
                scraped = await scrape_url(url, formats=["markdown"])

                if scraped.get("success"):
                    content = scraped.get("markdown", "")[:1000]  # First 1000 chars

                    documents.append({
                        "title": title,
                        "url": url,
                        "content": content,
                        "published_date": result.get("publishedDate", "Unknown"),
                        "relevance": 0.9 - (idx * 0.1)  # Simple relevance scoring
                    })

            return {
                "documents": documents,
                "total_found": len(documents),
                "search_query": query,
                "status": "success - MCP Firecrawl"
            }

    except Exception as e:
        # Fallback to mock data if MCP fails
        pass

    # Fallback to mock data
    return {
        "documents": [
            {
                "title": f"{query} 관련 환경부 문서",
                "url": "https://me.go.kr/home/web/main.do",
                "content": f"{query}에 대한 최신 환경부 규정 정보입니다. 정확한 내용은 환경부 공식 사이트를 참조하세요.",
                "published_date": "2024-11-15",
                "relevance": 0.75
            }
        ],
        "total_found": 1,
        "search_query": query,
        "status": "fallback - mock data (MCP unavailable)"
    }


async def krx_api_realtime(endpoint: str = "price") -> Dict[str, Any]:
    """Access KRX emission trading API for real-time data via MCP.

    Args:
        endpoint: API endpoint ("price", "volume", "trades")

    Returns:
        Real-time market data from KRX API
    """
    # TODO: Implement actual KRX API integration via MCP
    return await get_krx_market_data()


# ============================================================================
# 6. Skills Tools - Document Generation
# ============================================================================


async def generate_docx_report(
    report_type: str,
    company_data: Dict[str, Any],
    emission_data: Dict[str, Any]
) -> Dict[str, str]:
    """Generate DOCX document (business plan, consulting report).

    Uses python-docx library to generate professional documents.

    Args:
        report_type: Type of report ("business_plan", "consulting_report", "contract")
        company_data: Company information
        emission_data: Emission calculation results

    Returns:
        {
            "file_path": "/tmp/business_plan_ABC.docx",
            "file_size": "2.4 MB",
            "pages": 12,
            "status": "success"
        }
    """
    # TODO: Implement python-docx generation with Claude Skills integration
    return {
        "file_path": f"/tmp/{report_type}_{company_data.get('id', 'unknown')}.docx",
        "file_size": "2.4 MB",
        "pages": 12,
        "status": "mock_generated - python-docx 구현 예정"
    }


async def generate_xlsx_spreadsheet(
    spreadsheet_type: str,
    emission_data: Dict[str, Any]
) -> Dict[str, str]:
    """Generate XLSX spreadsheet (calculation sheet, tracking sheet).

    Uses openpyxl library to generate Excel spreadsheets with charts.

    Args:
        spreadsheet_type: Type of spreadsheet ("calculation", "tracking", "comparison")
        emission_data: Emission data to include

    Returns:
        {
            "file_path": "/tmp/emission_calc.xlsx",
            "file_size": "156 KB",
            "sheets": ["배출량 계산", "차트"],
            "status": "success"
        }
    """
    # TODO: Implement openpyxl generation with formatting and charts
    return {
        "file_path": f"/tmp/{spreadsheet_type}.xlsx",
        "file_size": "156 KB",
        "sheets": ["배출량 계산", "차트"],
        "status": "mock_generated - openpyxl 구현 예정"
    }


async def generate_pdf_form(
    form_type: str,
    form_data: Dict[str, Any]
) -> Dict[str, str]:
    """Generate PDF form (application, certificate).

    Uses reportlab or pypdf library to generate PDF forms.

    Args:
        form_type: Type of form ("application", "certificate", "contract")
        form_data: Form field data

    Returns:
        {
            "file_path": "/tmp/application_form.pdf",
            "file_size": "245 KB",
            "status": "success"
        }
    """
    # TODO: Implement PDF generation (reportlab or pypdf)
    return {
        "file_path": f"/tmp/{form_type}_form.pdf",
        "file_size": "245 KB",
        "status": "mock_generated - PDF 라이브러리 구현 예정"
    }


# ============================================================================
# 7. Artifact Tools - Interactive Visualizations
# ============================================================================


def generate_mermaid_chart(
    chart_type: str,
    data: Dict[str, Any]
) -> Dict[str, str]:
    """Generate Mermaid diagram code for visualization.

    Args:
        chart_type: Type of chart ("gantt", "pie", "flowchart")
        data: Chart data

    Returns:
        {
            "type": "mermaid",
            "code": "gantt\n    title 탄소 감축 로드맵\n    ...",
            "artifact_id": "mermaid_12345"
        }
    """
    mermaid_templates = {
        "gantt": """gantt
    title 탄소 감축 로드맵
    dateFormat YYYY-MM
    section Phase 1
    LED 교체 :2025-01, 3M
    section Phase 2
    태양광 설치 :2025-10, 8M""",
        "pie": """pie title Scope별 배출 비중
    "Scope 1" : {scope1}
    "Scope 2" : {scope2}
    "Scope 3" : {scope3}""",
        "flowchart": """flowchart TD
    A[배출량 계산] --> B{기준 초과?}
    B -->|Yes| C[감축 계획 수립]
    B -->|No| D[모니터링 지속]
    C --> E[배출권 거래 검토]"""
    }

    template = mermaid_templates.get(chart_type, mermaid_templates["flowchart"])
    code = template.format(**data) if "{" in template else template

    return {
        "type": "mermaid",
        "code": code,
        "artifact_id": f"mermaid_{int(time.time())}"
    }


def generate_react_component(
    component_type: str,
    data: Dict[str, Any]
) -> Dict[str, str]:
    """Generate React component code for interactive visualizations.

    Args:
        component_type: Type of component ("chart", "calculator", "simulator")
        data: Component data

    Returns:
        {
            "type": "react",
            "code": "import { BarChart, ... } from 'recharts'; ...",
            "artifact_id": "react_12345"
        }
    """
    if component_type == "chart":
        react_code = f'''import {{ BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer }} from 'recharts';

export default function EmissionChart() {{
  const data = [
    {{ name: 'Scope 1', value: {data.get('scope1', 0)}, fill: '#0D9488' }},
    {{ name: 'Scope 2', value: {data.get('scope2', 0)}, fill: '#14B8A6' }},
    {{ name: 'Scope 3', value: {data.get('scope3', 0)}, fill: '#5EEAD4' }}
  ];

  return (
    <div className="bg-white rounded-lg p-6 shadow-lg">
      <h3 className="text-xl font-bold text-hooxi-primary mb-4">
        📊 Scope별 배출량 (tCO2eq)
      </h3>

      <ResponsiveContainer width="100%" height={{300}}>
        <BarChart data={{data}}>
          <XAxis dataKey="name" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="value" radius={{[8, 8, 0, 0]}} />
        </BarChart>
      </ResponsiveContainer>

      <div className="mt-4 space-y-2">
        {{data.map(item => (
          <div key={{item.name}} className="flex justify-between">
            <span>{{item.name}}</span>
            <span className="font-bold">{{item.value}} tCO2eq</span>
          </div>
        ))}}
      </div>
    </div>
  );
}}'''

    elif component_type == "calculator":
        react_code = '''import { useState } from 'react';

export default function Calculator() {
  const [fuel, setFuel] = useState(0);
  const [elec, setElec] = useState(0);
  const [logistics, setLogistics] = useState(0);

  const FACTORS = {
    fuel: 2.31,
    electricity: 0.4781,
    logistics: 0.185
  };

  const scope1 = (fuel * FACTORS.fuel) / 1000;
  const scope2 = (elec * FACTORS.electricity) / 1000;
  const scope3 = (logistics * FACTORS.logistics) / 1000;
  const total = scope1 + scope2 + scope3;

  return (
    <div className="bg-white rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4">🧮 배출량 계산기</h3>

      <div className="space-y-3">
        <div>
          <label className="block text-sm font-medium mb-1">휘발유 (L)</label>
          <input
            type="number"
            value={fuel}
            onChange={(e) => setFuel(Number(e.target.value))}
            className="w-full px-3 py-2 border rounded-lg"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">전기 (kWh)</label>
          <input
            type="number"
            value={elec}
            onChange={(e) => setElec(Number(e.target.value))}
            className="w-full px-3 py-2 border rounded-lg"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">물류 (km)</label>
          <input
            type="number"
            value={logistics}
            onChange={(e) => setLogistics(Number(e.target.value))}
            className="w-full px-3 py-2 border rounded-lg"
          />
        </div>
      </div>

      <div className="mt-6 p-4 bg-teal-50 rounded-lg">
        <h4 className="font-bold mb-2">계산 결과</h4>
        <div className="space-y-1 text-sm">
          <div>Scope 1: {scope1.toFixed(2)} tCO2eq</div>
          <div>Scope 2: {scope2.toFixed(2)} tCO2eq</div>
          <div>Scope 3: {scope3.toFixed(2)} tCO2eq</div>
          <div className="pt-2 mt-2 border-t font-bold text-lg">
            총: {total.toFixed(2)} tCO2eq
          </div>
        </div>
      </div>
    </div>
  );
}'''

    else:  # simulator
        react_code = f'''import {{ useState }} from 'react';

export default function TradingSimulator() {{
  const [targetPrice, setTargetPrice] = useState(14250);
  const [targetVolume, setTargetVolume] = useState(1000);

  const matchScore = Math.min(100, Math.floor((targetVolume / 10000) * 100));

  return (
    <div className="bg-white rounded-lg p-6">
      <h3 className="text-xl font-bold mb-4">💹 거래 매칭 시뮬레이터</h3>

      <div className="space-y-3 mb-6">
        <div>
          <label className="block text-sm font-medium mb-1">목표 가격 (원/톤)</label>
          <input
            type="number"
            value={{targetPrice}}
            onChange={{(e) => setTargetPrice(Number(e.target.value))}}
            className="w-full px-3 py-2 border rounded-lg"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">거래량 (톤)</label>
          <input
            type="number"
            value={{targetVolume}}
            onChange={{(e) => setTargetVolume(Number(e.target.value))}}
            className="w-full px-3 py-2 border rounded-lg"
          />
        </div>
      </div>

      <div className="p-4 bg-teal-50 rounded-lg">
        <h4 className="font-bold mb-2">AI 매칭 점수</h4>
        <div className="text-3xl font-bold text-teal-600">{{matchScore}}/100</div>
        <div className="mt-2 text-sm">
          {{matchScore > 70 ? '✅ 매칭 가능성 높음' : '⚠️ 조건 조정 필요'}}
        </div>
      </div>
    </div>
  );
}}'''

    return {
        "type": "react",
        "code": react_code,
        "artifact_id": f"react_{int(time.time())}"
    }


# ============================================================================
# Legacy tool (keep for backward compatibility)
# ============================================================================


async def search(query: str) -> Optional[dict[str, Any]]:
    """Search for general web results using Tavily.

    This is a legacy tool kept for backward compatibility.
    For emission-specific searches, use search_emission_database instead.
    """
    configuration = Configuration.from_context()
    wrapped = TavilySearch(max_results=configuration.max_search_results)
    return cast(dict[str, Any], await wrapped.ainvoke({"query": query}))


# ============================================================================
# Tool Registry
# ============================================================================

TOOLS: List[Callable[..., Any]] = [
    # Core Tools (4가지)
    search_emission_database,  # 1. SearchTool (ChromaDB RAG)
    calculate_emissions,  # 2. CalculatorTool (Scope 1/2/3 - 내부 계산)
    get_krx_market_data,  # 3. MarketTool (KRX - MCP)
    ai_matching_score,  # 3-2. MarketTool AI 매칭
    request_human_consultation,  # 4. ConsultationTool (Naver Works - MCP)

    # MCP Tools
    firecrawl_government_docs,  # Firecrawl MCP
    krx_api_realtime,  # KRX API MCP

    # Skills Tools (문서 생성)
    generate_docx_report,  # DOCX
    generate_xlsx_spreadsheet,  # XLSX
    generate_pdf_form,  # PDF

    # Artifact Tools (시각화)
    generate_mermaid_chart,  # Mermaid
    generate_react_component,  # React

    # Legacy
    search,
]
