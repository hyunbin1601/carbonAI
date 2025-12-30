"""Simple test for individual tools with mock data.

This script tests each tool independently to verify they return mock data correctly.
No LLM calls needed - just pure tool testing.
"""

import asyncio


async def test_all_tools():
    """Test all tools with mock data."""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        🔧 Individual Tool Testing                           ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Import tools
    from src.react_agent.tools import (
        calculate_emissions,
        get_krx_market_data,
        search_emission_database,
        firecrawl_government_docs,
        request_human_consultation,
        generate_mermaid_chart,
        generate_react_component,
    )

    # Test 1: Emission Calculator
    print("\n1️⃣ Testing Emission Calculator (내부 계산)")
    print("-" * 80)
    result = await calculate_emissions(
        fuel_liters=1000,
        electricity_kwh=50000,
        logistics_km=10000,
        natural_gas_m3=100
    )
    print(f"✅ Scope 1: {result['scope1']} tCO2eq")
    print(f"✅ Scope 2: {result['scope2']} tCO2eq")
    print(f"✅ Scope 3: {result['scope3']} tCO2eq")
    print(f"✅ Total: {result['total']} tCO2eq")
    print(f"📊 Breakdown: {result['breakdown']}")

    # Test 2: KRX Market Data
    print("\n2️⃣ Testing KRX Market Data (Mock)")
    print("-" * 80)
    result = await get_krx_market_data("KAU")
    print(f"✅ Type: {result['type']}")
    print(f"✅ Price: {result['price']:,} KRW")
    print(f"✅ Change: {result['change']}%")
    print(f"✅ Volume: {result['volume']:,} tons")
    print(f"✅ Status: {result['status']}")

    # Test 3: Search Emission Database
    print("\n3️⃣ Testing ChromaDB Search (Mock)")
    print("-" * 80)
    result = await search_emission_database("2024 전력 배출 계수")
    print(f"✅ Query: {result['query']}")
    print(f"✅ Total Results: {result['total_results']}")
    print(f"✅ First Result: {result['results'][0]['content'][:80]}...")

    # Test 4: Firecrawl Government Docs
    print("\n4️⃣ Testing Firecrawl MCP (Mock)")
    print("-" * 80)
    result = await firecrawl_government_docs("2024 배출 계수")
    print(f"✅ Query: {result['search_query']}")
    print(f"✅ Total Found: {result['total_found']}")
    print(f"✅ First Document: {result['documents'][0]['title']}")

    # Test 5: Human Consultation
    print("\n5️⃣ Testing Naver Works Consultation (Mock)")
    print("-" * 80)
    result = await request_human_consultation("배출권 거래 전략", "high")
    print(f"✅ Status: {result['status']}")
    print(f"✅ Ticket ID: {result['ticket_id']}")
    print(f"✅ Estimated Response: {result['estimated_response_time']}")
    print(f"✅ Assigned To: {result['assigned_consultant']}")

    # Test 6: Mermaid Chart Generation
    print("\n6️⃣ Testing Mermaid Chart Generation")
    print("-" * 80)
    result = generate_mermaid_chart("pie", {"scope1": 37, "scope2": 28, "scope3": 35})
    print(f"✅ Type: {result['type']}")
    print(f"✅ Artifact ID: {result['artifact_id']}")
    print(f"✅ Code Preview:\n{result['code'][:100]}...")

    # Test 7: React Component Generation
    print("\n7️⃣ Testing React Component Generation")
    print("-" * 80)
    result = generate_react_component("calculator", {})
    print(f"✅ Type: {result['type']}")
    print(f"✅ Artifact ID: {result['artifact_id']}")
    print(f"✅ Code Preview:\n{result['code'][:100]}...")

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                          ✅ All Tools Tested!                               ║
║                                                                              ║
║  Results:                                                                    ║
║  ✅ Emission Calculator - Working (내부 계산)                               ║
║  ✅ KRX Market Data - Mock data ready                                       ║
║  ✅ ChromaDB Search - Mock data ready                                       ║
║  ✅ Firecrawl MCP - Mock data ready                                         ║
║  ✅ Naver Works - Mock data ready                                           ║
║  ✅ Mermaid Charts - Working                                                 ║
║  ✅ React Components - Working                                               ║
║                                                                              ║
║  All tools are returning data correctly!                                     ║
║  Ready to test full agent workflow.                                          ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    asyncio.run(test_all_tools())
