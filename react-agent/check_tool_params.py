"""MCP 도구 파라미터 확인"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from react_agent.sse_mcp_client import SSEMCPClient
import json


async def check_tool_params():
    """get_company_id_by_name 도구의 파라미터 확인"""

    client = SSEMCPClient(base_url="https://hooxi.shinssy.com")

    try:
        await client.initialize()
        tools = await client.list_tools()

        # get_company_id_by_name 도구 찾기
        target_tool = None
        for tool in tools:
            if tool.get("name") == "get_company_id_by_name":
                target_tool = tool
                break

        if target_tool:
            print("=" * 80)
            print("get_company_id_by_name 도구 정의:")
            print("=" * 80)
            print(json.dumps(target_tool, indent=2, ensure_ascii=False))
            print("\n" + "=" * 80)

            # inputSchema 확인
            input_schema = target_tool.get("inputSchema", {})
            properties = input_schema.get("properties", {})
            required = input_schema.get("required", [])

            print("\n파라미터 정의:")
            for param_name, param_info in properties.items():
                is_required = "(필수)" if param_name in required else "(선택)"
                print(f"  - {param_name}: {param_info.get('type')} {is_required}")
                print(f"    설명: {param_info.get('description', 'N/A')}")

            # 테스트 호출
            print("\n" + "=" * 80)
            print("테스트 호출: 후시파트너스111")
            print("=" * 80)

            # 올바른 파라미터 이름으로 호출
            for param_name in properties.keys():
                print(f"\n{param_name} 파라미터로 호출 시도...")
                try:
                    result = await client.call_tool(
                        "get_company_id_by_name",
                        {param_name: "후시파트너스111"}
                    )
                    print(f"성공! 결과:")
                    print(json.dumps(result, indent=2, ensure_ascii=False))
                    break
                except Exception as e:
                    print(f"실패: {e}")

        else:
            print("get_company_id_by_name 도구를 찾을 수 없습니다.")

    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(check_tool_params())
