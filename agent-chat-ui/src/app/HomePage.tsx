"use client";

import React, { useState } from "react";
import { ChatBanner } from "@/components/chat-banner";
import { FloatingChat } from "@/components/floating-chat";
import { ChatConfig } from "@/lib/config";

interface HomePageProps {
  initialConfig: ChatConfig;
}

export function HomePage({ initialConfig }: HomePageProps) {
  const [isChatOpen, setIsChatOpen] = useState(false);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-teal-50/30 to-emerald-50/30">
      {/* Main Content Area */}
      <div className="min-h-screen pb-32">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12 sm:py-16 lg:py-24">
          <div className="text-center">
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-bold text-gray-900 mb-6">
              {initialConfig.branding.appName}
            </h1>
            {initialConfig.branding.description && (
              <p className="text-lg sm:text-xl text-gray-600 mb-8 max-w-2xl mx-auto">
                {initialConfig.branding.description}
              </p>
            )}
          </div>
        </div>
      </div>

      {/* Chat Banner */}
      <ChatBanner onClick={() => setIsChatOpen(true)} />

      {/* Floating Chat */}
      <FloatingChat
        isOpen={isChatOpen}
        onClose={() => setIsChatOpen(false)}
        initialConfig={initialConfig}
      />
    </div>
  );
}

