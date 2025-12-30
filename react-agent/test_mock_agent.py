"""Test script for Hooxi Agent with mock data.

This script tests the 5-node agent architecture using mock data
to verify the system works correctly before integrating real APIs.
"""

import asyncio
from langchain_core.messages import HumanMessage

from src.react_agent.graph import graph


async def test_agent(user_message: str):
    """Test the agent with a user message.

    Args:
        user_message: User's question or request
    """
    print(f"\n{'='*80}")
    print(f"🧪 Testing: {user_message}")
    print(f"{'='*80}\n")

    # Prepare input
    input_state = {
        "messages": [HumanMessage(content=user_message)]
    }

    # Run the agent
    result = await graph.ainvoke(input_state)

    # Print results
    print("\n📊 Results:")
    print(f"Intent Type: {result.get('intent_type', 'N/A')}")
    print(f"Selected Tools: {', '.join(result.get('selected_tools', []))}")
    print(f"Quality Score: {result.get('quality_score', 'N/A')}/100")
    print(f"Verification Feedback: {result.get('verification_feedback', 'N/A')}")
    print(f"Regeneration Count: {result.get('regeneration_count', 0)}")

    # Print final response
    print("\n💬 Final Response:")
    if result.get('messages'):
        for msg in result['messages']:
            if hasattr(msg, 'content'):
                print(msg.content)

    print(f"\n{'='*80}\n")


async def main():
    """Run all test scenarios."""

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                   🚀 Hooxi Agent Mock Data Test Suite                       ║
║                                                                              ║
║  Testing 5-Node Architecture:                                               ║
║  [Classify (Sonnet)] → [Route] → [Execute] → [Generate (Haiku)] → [Verify] ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)

    # Test scenarios
    test_cases = [
        # 1. MEASUREMENT - Emission calculation
        {
            "message": "휘발유 1000리터, 전기 50000kWh, 물류 10000km 사용했을 때 배출량 계산해줘",
            "expected_intent": "MEASUREMENT",
            "description": "배출량 계산 테스트"
        },

        # 2. TRADING - KRX market data
        {
            "message": "KAU 현재 시세 알려줘",
            "expected_intent": "TRADING",
            "description": "KRX 시세 조회 테스트"
        },

        # 3. FAQ - Emission factors
        {
            "message": "2024년 전력 배출 계수가 뭐야?",
            "expected_intent": "FAQ",
            "description": "FAQ 검색 테스트"
        },

        # 4. REPORTING - Document generation
        {
            "message": "사업계획서 만들어줘",
            "expected_intent": "REPORTING",
            "description": "문서 생성 테스트"
        },

        # 5. CONSULTATION - Human consultant
        {
            "message": "전문가 상담 연결해줘",
            "expected_intent": "CONSULTATION",
            "description": "상담 연결 테스트"
        },
    ]

    # Run tests
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}/5: {test_case['description']}")
        print(f"Expected Intent: {test_case['expected_intent']}")

        try:
            await test_agent(test_case['message'])
            print("✅ Test completed successfully")
        except Exception as e:
            print(f"❌ Test failed with error: {e}")
            import traceback
            traceback.print_exc()

        # Pause between tests
        if i < len(test_cases):
            print("\nPress Enter to continue to next test...")
            # input()  # Uncomment to pause between tests

    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║                          ✅ All Tests Completed!                            ║
║                                                                              ║
║  Next steps:                                                                 ║
║  1. Review the quality scores and responses                                 ║
║  2. Integrate real APIs (ChromaDB, KRX, Firecrawl)                          ║
║  3. Implement document generation (python-docx, openpyxl)                   ║
║  4. Add artifact rendering in frontend                                      ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    # Run the async main function
    asyncio.run(main())
