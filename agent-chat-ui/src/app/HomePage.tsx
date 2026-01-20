"use client";

import React, { useState } from "react";
import dynamic from "next/dynamic";
import { FloatingChat } from "@/components/floating-chat";
import { ChatConfig } from "@/lib/config";
import { Rocket } from "lucide-react";

// Spline을 클라이언트 전용으로 동적 로드 (SSR 비활성화)
const Spline = dynamic(
  () => import("@splinetool/react-spline").then((mod) => mod.default),
  {
    ssr: false,
    loading: () => (
      <div className="w-full h-full bg-gradient-to-br from-slate-900 via-teal-900 to-emerald-900 flex items-center justify-center">
        <div className="text-white/60 text-lg animate-pulse">Loading 3D Scene...</div>
      </div>
    ),
  }
);

interface HomePageProps {
  initialConfig: ChatConfig;
}

export function HomePage({ initialConfig }: HomePageProps) {
  const [isChatOpen, setIsChatOpen] = useState(false);

  return (
    <div className="relative w-full h-screen overflow-hidden">
      {/* Spline 3D 배경 */}
      <div className="absolute inset-0 z-0">
        <Spline scene="https://prod.spline.design/2YTWwBKDt94Jth9t/scene.splinecode" />
      </div>

      {/* 오버레이 콘텐츠 */}
      <div className="relative z-10 flex flex-col items-center justify-center h-full px-4 pointer-events-none">
        {/* 로고 및 타이틀 */}
        <div className="text-center mb-8">
          <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-white mb-4 drop-shadow-lg">
            {initialConfig.branding.appName || "CarbonAI"}
          </h1>
          {initialConfig.branding.description && (
            <p className="text-lg sm:text-xl text-white/80 max-w-2xl mx-auto drop-shadow-md">
              {initialConfig.branding.description}
            </p>
          )}
        </div>

        {/* Start 버튼 */}
        <button
          onClick={() => setIsChatOpen(true)}
          className="pointer-events-auto group relative px-10 py-5 bg-gradient-to-r from-teal-500 to-emerald-500 hover:from-teal-400 hover:to-emerald-400 text-white text-xl font-bold rounded-full shadow-2xl hover:shadow-teal-500/50 transition-all duration-300 transform hover:scale-105 active:scale-95"
        >
          <span className="flex items-center gap-3">
            <Rocket className="w-6 h-6 group-hover:animate-bounce" />
            Start!
          </span>
          {/* 버튼 글로우 효과 */}
          <div className="absolute inset-0 rounded-full bg-gradient-to-r from-teal-400 to-emerald-400 opacity-0 group-hover:opacity-30 blur-xl transition-opacity duration-300" />
        </button>

        {/* 하단 힌트 텍스트 */}
        <p className="mt-8 text-white/60 text-sm">
          AI 탄소배출 전문 상담을 시작하세요
        </p>
      </div>

      {/* Floating Chat */}
      <FloatingChat
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
        initialConfig={initialConfig}
      />
    </div>
  );
}
