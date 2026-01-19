"use client";

import { useEffect, useRef, useState } from "react";
import * as agCharts from "ag-charts-community";
import type { AgChartOptions } from "ag-charts-community";
import { cn } from "@/lib/utils";

interface AGChartProps {
  config: string | AgChartOptions;
  className?: string;
}

export function AGChart({ config, className }: AGChartProps) {
  const chartRef = useRef<HTMLDivElement>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    if (!chartRef.current) return;

    setIsLoading(true);
    setError(null);

    let chart: any;

    try {
      let options: AgChartOptions;

      if (typeof config === "string") {
        // JSON 문자열을 파싱
        options = JSON.parse(config);
      } else {
        options = config;
      }

      // 기본 테마 및 스타일 적용
      const defaultOptions: AgChartOptions = {
        container: chartRef.current,
        background: {
          fill: "transparent",
        },
        ...options,
      };

      // AG Charts 직접 생성
      chart = agCharts.AgCharts.create(defaultOptions);
      setIsLoading(false);
    } catch (err) {
      console.error("AG Chart parsing error:", err);
      setError(err instanceof Error ? err.message : "차트를 렌더링할 수 없습니다.");
      setIsLoading(false);
    }

    // Cleanup
    return () => {
      if (chart) {
        chart.destroy();
      }
    };
  }, [config]);

  if (error) {
    return (
      <div
        className={cn(
          "rounded-xl bg-muted/50 dark:bg-zinc-900 p-4 border border-red-500/30",
          className
        )}
      >
        <p className="text-sm text-red-500">오류: {error}</p>
        <pre className="mt-2 text-xs text-muted-foreground overflow-x-auto">
          {typeof config === "string" ? config : JSON.stringify(config, null, 2)}
        </pre>
      </div>
    );
  }

  return (
    <div
      className={cn(
        "ag-chart-container rounded-xl bg-muted/50 dark:bg-zinc-900 p-4 border border-border/30 dark:border-zinc-700",
        className
      )}
    >
      {isLoading && (
        <div className="text-center text-sm text-muted-foreground py-4">
          차트 렌더링 중...
        </div>
      )}
      <div ref={chartRef} className="w-full h-[400px]" />
    </div>
  );
}
