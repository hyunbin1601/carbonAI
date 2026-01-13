import React, { useState, useRef, useEffect } from 'react';

const HooxiBotPro = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [isSidebarMode, setIsSidebarMode] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showWelcome, setShowWelcome] = useState(true);
  const messagesEndRef = useRef(null);

  const welcomeOptions = [
    { 
      icon: '📊', 
      text: '탄소 배출량 측정 서비스',
      desc: 'Scope 1, 2, 3 측정부터 보고까지',
      value: '탄소 배출량을 어떻게 측정하나요?',
      color: 'from-emerald-500 to-teal-500'
    },
    { 
      icon: '💰', 
      text: '배출권을 판매하고 싶어요',
      desc: '보유 배출권 수익화 전략',
      value: '배출권 판매 절차를 알려주세요',
      color: 'from-blue-500 to-cyan-500'
    },
    { 
      icon: '🛒', 
      text: '배출권을 구매하고 싶어요',
      desc: '규제 대응 및 리스크 관리',
      value: '배출권 구매 방법을 알고 싶어요',
      color: 'from-violet-500 to-purple-500'
    },
    { 
      icon: '👤', 
      text: '상담원과 연결해주세요',
      desc: '전문가 1:1 맞춤 상담',
      value: '상담원 연결을 요청합니다',
      color: 'from-orange-500 to-amber-500'
    }
  ];

  const quickReplies = [
    { text: '배출권 거래 시작하기', icon: '🚀' },
    { text: '탄소 배출량 계산', icon: '🧮' },
    { text: '상담 예약하기', icon: '📅' }
  ];

  const addMessage = (text, role = 'user') => {
    const newMessage = {
      id: Date.now(),
      text,
      role,
      timestamp: new Date().toISOString()
    };
    setMessages(prev => [...prev, newMessage]);
    setShowWelcome(false);
  };

  const simulateBotResponse = (userMessage) => {
    setIsTyping(true);
    
    setTimeout(() => {
      let botResponse = '';
      
      if (userMessage.includes('측정') || userMessage.includes('배출량')) {
        botResponse = '탄소 배출량 측정 서비스에 관심 가져주셔서 감사합니다! 🌱\n\n후시파트너스는 기업의 탄소 배출량을 정확하게 측정하고 관리할 수 있는 솔루션을 제공합니다.\n\n• Scope 1: 직접 배출\n• Scope 2: 간접 배출 (전기, 열)\n• Scope 3: 기타 간접 배출\n\n모든 배출원을 체계적으로 추적하실 수 있습니다. 자세한 상담을 원하시나요?';
      } else if (userMessage.includes('판매')) {
        botResponse = '배출권 판매를 고려하고 계시군요! 💰\n\n후시파트너스의 NetZ 플랫폼을 통해 보유하신 배출권을 효율적으로 거래하실 수 있습니다.\n\n현재 시장 가격은 KAU 기준 약 8,000~12,000원 수준입니다. 보유하고 계신 배출권의 종류와 수량을 알려주시면 더 구체적인 안내가 가능합니다.';
      } else if (userMessage.includes('구매')) {
        botResponse = '배출권 구매를 원하시는군요! 🛒\n\n후시파트너스는 투명하고 안전한 배출권 거래 플랫폼을 운영하고 있습니다.\n\n구매 프로세스:\n1. 수요량 산정\n2. 시장 분석\n3. 최적가 매칭\n4. 안전한 거래 체결\n\n필요하신 배출권의 종류와 수량을 알려주시면 최적의 거래를 중개해드리겠습니다.';
      } else if (userMessage.includes('상담')) {
        botResponse = '전문 상담원 연결을 도와드리겠습니다! 📞\n\n성함과 연락처, 그리고 상담 희망 시간대를 알려주시면 빠른 시일 내에 연락드리겠습니다.\n\n상담 가능 시간:\n• 평일 09:00 - 18:00\n• 점심시간 제외 (12:00 - 13:00)';
      } else {
        botResponse = '안녕하세요! 후시봇입니다. 🌱\n\n탄소배출 관리와 배출권 거래에 대해 궁금하신 점을 자유롭게 물어보세요. 전문적이고 친절하게 도와드리겠습니다!';
      }
      
      setIsTyping(false);
      addMessage(botResponse, 'assistant');
    }, 1800);
  };

  const handleSendMessage = (text = inputValue) => {
    if (!text.trim()) return;
    
    addMessage(text, 'user');
    setInputValue('');
    simulateBotResponse(text);
  };

  const handleWelcomeOption = (option) => {
    handleSendMessage(option.value);
  };

  const handleQuickReply = (reply) => {
    handleSendMessage(reply.text);
  };

  const handleNewChat = () => {
    setMessages([]);
    setShowWelcome(true);
    setInputValue('');
  };

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  return (
    <>
      <style>{`
        @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');
        
        * {
          margin: 0;
          padding: 0;
          box-sizing: border-box;
        }
        
        body {
          font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: linear-gradient(135deg, #f8fafc 0%, #e0f2f1 100%);
          color: #1a1a1a;
          line-height: 1.6;
          overflow-x: hidden;
        }
        
        /* 배경 패턴 효과 */
        body::before {
          content: '';
          position: fixed;
          top: 0;
          left: 0;
          width: 100%;
          height: 100%;
          background-image: 
            radial-gradient(circle at 20% 30%, rgba(13, 148, 136, 0.03) 0%, transparent 50%),
            radial-gradient(circle at 80% 70%, rgba(13, 148, 136, 0.03) 0%, transparent 50%);
          pointer-events: none;
          z-index: -1;
        }
        
        /* 플로팅 버튼 - 프리미엄 디자인 */
        .hooxi-float-button {
          position: fixed;
          bottom: 28px;
          right: 28px;
          width: 68px;
          height: 68px;
          border-radius: 50%;
          background: linear-gradient(135deg, #0D9488 0%, #14B8A6 100%);
          border: 3px solid rgba(255, 255, 255, 0.95);
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          box-shadow: 
            0 8px 32px rgba(13, 148, 136, 0.3),
            0 0 0 0 rgba(13, 148, 136, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
          transition: all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
          z-index: 9998;
          animation: float 3s ease-in-out infinite;
        }
        
        @keyframes float {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-8px); }
        }
        
        .hooxi-float-button:hover {
          transform: scale(1.1) translateY(-4px);
          box-shadow: 
            0 12px 48px rgba(13, 148, 136, 0.4),
            0 0 0 8px rgba(13, 148, 136, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
          animation: none;
        }
        
        .hooxi-float-button:active {
          transform: scale(1.02);
        }
        
        .hooxi-float-button svg {
          width: 32px;
          height: 32px;
          stroke: white;
          stroke-width: 2.5;
          filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.1));
          transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        
        .hooxi-float-button.open svg {
          transform: rotate(90deg);
        }
        
        /* 알림 배지 */
        .hooxi-float-button::after {
          content: '';
          position: absolute;
          top: 2px;
          right: 2px;
          width: 14px;
          height: 14px;
          background: linear-gradient(135deg, #10B981 0%, #059669 100%);
          border-radius: 50%;
          border: 2px solid white;
          animation: pulse-badge 2s ease-in-out infinite;
        }
        
        @keyframes pulse-badge {
          0%, 100% { transform: scale(1); opacity: 1; }
          50% { transform: scale(1.2); opacity: 0.8; }
        }
        
        .hooxi-float-button.open::after {
          display: none;
        }
        
        /* 채팅 윈도우 - 프리미엄 글래스모피즘 */
        .hooxi-chat-window {
          position: fixed;
          background: rgba(255, 255, 255, 0.98);
          backdrop-filter: blur(20px) saturate(180%);
          -webkit-backdrop-filter: blur(20px) saturate(180%);
          border: 1px solid rgba(13, 148, 136, 0.1);
          box-shadow: 
            0 24px 64px rgba(13, 148, 136, 0.15),
            0 0 0 1px rgba(255, 255, 255, 0.5),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
          display: flex;
          flex-direction: column;
          z-index: 9999;
          transition: all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1);
          opacity: 0;
          pointer-events: none;
        }
        
        .hooxi-chat-window.open {
          opacity: 1;
          pointer-events: all;
        }
        
        /* 플로팅 모드 */
        .hooxi-chat-window.floating {
          bottom: 116px;
          right: 28px;
          width: 420px;
          height: 640px;
          max-height: calc(100vh - 160px);
          border-radius: 24px;
          transform: translateY(40px) scale(0.9);
        }
        
        .hooxi-chat-window.floating.open {
          transform: translateY(0) scale(1);
        }
        
        /* 사이드바 모드 */
        .hooxi-chat-window.sidebar {
          top: 0;
          right: 0;
          width: 440px;
          height: 100vh;
          border-radius: 0;
          border-right: none;
          transform: translateX(100%);
        }
        
        .hooxi-chat-window.sidebar.open {
          transform: translateX(0);
        }
        
        /* 헤더 - 프리미엄 그라디언트 */
        .hooxi-header {
          background: linear-gradient(135deg, #0F766E 0%, #0D9488 50%, #14B8A6 100%);
          color: white;
          padding: 24px 28px;
          display: flex;
          align-items: center;
          justify-content: space-between;
          flex-shrink: 0;
          position: relative;
          overflow: hidden;
        }
        
        .hooxi-header::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: 
            radial-gradient(circle at 30% 50%, rgba(255, 255, 255, 0.1) 0%, transparent 50%),
            radial-gradient(circle at 70% 80%, rgba(255, 255, 255, 0.05) 0%, transparent 50%);
          pointer-events: none;
        }
        
        .hooxi-header-left {
          display: flex;
          align-items: center;
          gap: 14px;
          position: relative;
          z-index: 1;
        }
        
        .hooxi-header-icon {
          font-size: 32px;
          line-height: 1;
          animation: gentle-bounce 2s ease-in-out infinite;
        }
        
        @keyframes gentle-bounce {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-3px); }
        }
        
        .hooxi-header-title-wrap {
          display: flex;
          flex-direction: column;
          gap: 2px;
        }
        
        .hooxi-header-title {
          font-size: 18px;
          font-weight: 700;
          line-height: 1.2;
          letter-spacing: -0.02em;
        }
        
        .hooxi-header-subtitle {
          font-size: 12px;
          opacity: 0.85;
          font-weight: 500;
        }
        
        .hooxi-header-buttons {
          display: flex;
          gap: 10px;
          position: relative;
          z-index: 1;
        }
        
        .hooxi-header-button {
          width: 40px;
          height: 40px;
          border: none;
          background: rgba(255, 255, 255, 0.15);
          backdrop-filter: blur(10px);
          border-radius: 10px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
          position: relative;
        }
        
        .hooxi-header-button:hover {
          background: rgba(255, 255, 255, 0.25);
          transform: translateY(-2px) scale(1.05);
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        
        .hooxi-header-button:active {
          transform: translateY(0) scale(0.98);
        }
        
        .hooxi-header-button svg {
          width: 20px;
          height: 20px;
          stroke: white;
          stroke-width: 2.5;
        }
        
        /* 툴팁 - 개선된 디자인 */
        .hooxi-header-button::after {
          content: attr(data-tooltip);
          position: absolute;
          bottom: -42px;
          left: 50%;
          transform: translateX(-50%) scale(0.9);
          background: rgba(26, 26, 26, 0.95);
          backdrop-filter: blur(10px);
          color: white;
          padding: 6px 12px;
          border-radius: 8px;
          font-size: 11px;
          font-weight: 600;
          white-space: nowrap;
          opacity: 0;
          pointer-events: none;
          transition: all 0.2s cubic-bezier(0.34, 1.56, 0.64, 1);
          z-index: 10000;
          box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        }
        
        .hooxi-header-button:hover::after {
          opacity: 1;
          transform: translateX(-50%) scale(1);
        }
        
        /* 메시지 영역 - 프리미엄 그라디언트 배경 */
        .hooxi-messages {
          flex: 1;
          overflow-y: auto;
          padding: 28px;
          display: flex;
          flex-direction: column;
          gap: 20px;
          background: 
            linear-gradient(180deg, 
              rgba(248, 250, 252, 0.4) 0%, 
              rgba(224, 242, 241, 0.4) 50%,
              rgba(240, 253, 250, 0.4) 100%
            );
          position: relative;
        }
        
        .hooxi-messages::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background-image: 
            radial-gradient(circle at 25% 25%, rgba(13, 148, 136, 0.02) 0%, transparent 50%),
            radial-gradient(circle at 75% 75%, rgba(13, 148, 136, 0.02) 0%, transparent 50%);
          pointer-events: none;
        }
        
        .hooxi-messages::-webkit-scrollbar {
          width: 8px;
        }
        
        .hooxi-messages::-webkit-scrollbar-track {
          background: transparent;
        }
        
        .hooxi-messages::-webkit-scrollbar-thumb {
          background: linear-gradient(180deg, rgba(13, 148, 136, 0.3) 0%, rgba(13, 148, 136, 0.15) 100%);
          border-radius: 4px;
          border: 2px solid transparent;
          background-clip: padding-box;
        }
        
        .hooxi-messages::-webkit-scrollbar-thumb:hover {
          background: linear-gradient(180deg, rgba(13, 148, 136, 0.5) 0%, rgba(13, 148, 136, 0.3) 100%);
          background-clip: padding-box;
        }
        
        /* 웰컴 화면 - 프리미엄 디자인 */
        .hooxi-welcome {
          display: flex;
          flex-direction: column;
          align-items: center;
          text-align: center;
          padding: 48px 24px 24px;
          gap: 32px;
          position: relative;
          z-index: 1;
        }
        
        .hooxi-welcome-icon-wrap {
          position: relative;
        }
        
        .hooxi-welcome-icon {
          width: 88px;
          height: 88px;
          background: linear-gradient(135deg, rgba(13, 148, 136, 0.08) 0%, rgba(20, 184, 166, 0.08) 100%);
          backdrop-filter: blur(10px);
          border-radius: 24px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 44px;
          position: relative;
          box-shadow: 
            0 8px 32px rgba(13, 148, 136, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.5);
          animation: welcome-float 3s ease-in-out infinite;
        }
        
        @keyframes welcome-float {
          0%, 100% { transform: translateY(0px) rotate(0deg); }
          50% { transform: translateY(-8px) rotate(5deg); }
        }
        
        .hooxi-welcome-icon::before {
          content: '';
          position: absolute;
          inset: -4px;
          border-radius: 26px;
          background: linear-gradient(135deg, rgba(13, 148, 136, 0.2), rgba(20, 184, 166, 0.2));
          opacity: 0;
          animation: icon-pulse 2s ease-in-out infinite;
        }
        
        @keyframes icon-pulse {
          0%, 100% { opacity: 0; transform: scale(0.9); }
          50% { opacity: 1; transform: scale(1); }
        }
        
        .hooxi-welcome h2 {
          font-size: 24px;
          font-weight: 700;
          color: #1a1a1a;
          margin-bottom: 8px;
          letter-spacing: -0.02em;
          background: linear-gradient(135deg, #0F766E 0%, #14B8A6 100%);
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        }
        
        .hooxi-welcome p {
          font-size: 15px;
          color: #6B7280;
          line-height: 1.7;
          font-weight: 500;
        }
        
        .hooxi-welcome-options {
          width: 100%;
          display: flex;
          flex-direction: column;
          gap: 12px;
          margin-top: 8px;
        }
        
        .hooxi-welcome-option {
          width: 100%;
          padding: 18px 20px;
          background: rgba(255, 255, 255, 0.8);
          backdrop-filter: blur(10px);
          border: 1.5px solid rgba(13, 148, 136, 0.1);
          border-radius: 16px;
          cursor: pointer;
          transition: all 0.3s cubic-bezier(0.34, 1.56, 0.64, 1);
          display: flex;
          align-items: center;
          gap: 16px;
          text-align: left;
          position: relative;
          overflow: hidden;
        }
        
        .hooxi-welcome-option::before {
          content: '';
          position: absolute;
          top: 0;
          left: 0;
          right: 0;
          bottom: 0;
          background: linear-gradient(135deg, rgba(13, 148, 136, 0.05) 0%, rgba(20, 184, 166, 0.05) 100%);
          opacity: 0;
          transition: opacity 0.3s ease;
        }
        
        .hooxi-welcome-option:hover {
          border-color: rgba(13, 148, 136, 0.3);
          transform: translateY(-4px) scale(1.02);
          box-shadow: 
            0 12px 32px rgba(13, 148, 136, 0.15),
            0 0 0 1px rgba(13, 148, 136, 0.1);
        }
        
        .hooxi-welcome-option:hover::before {
          opacity: 1;
        }
        
        .hooxi-welcome-option:active {
          transform: translateY(-2px) scale(0.99);
        }
        
        .hooxi-welcome-option-icon {
          width: 52px;
          height: 52px;
          background: linear-gradient(135deg, rgba(13, 148, 136, 0.1) 0%, rgba(20, 184, 166, 0.1) 100%);
          border-radius: 14px;
          display: flex;
          align-items: center;
          justify-content: center;
          font-size: 26px;
          flex-shrink: 0;
          position: relative;
          z-index: 1;
          box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.5);
        }
        
        .hooxi-welcome-option-content {
          flex: 1;
          position: relative;
          z-index: 1;
        }
        
        .hooxi-welcome-option-title {
          font-size: 15px;
          font-weight: 600;
          color: #1a1a1a;
          margin-bottom: 4px;
          letter-spacing: -0.01em;
        }
        
        .hooxi-welcome-option-desc {
          font-size: 12px;
          color: #6B7280;
          font-weight: 500;
          line-height: 1.4;
        }
        
        /* 메시지 버블 - 프리미엄 디자인 */
        .hooxi-message {
          display: flex;
          flex-direction: column;
          max-width: 88%;
          animation: messageSlideIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1);
        }
        
        @keyframes messageSlideIn {
          from { 
            opacity: 0; 
            transform: translateY(20px) scale(0.95);
          }
          to { 
            opacity: 1; 
            transform: translateY(0) scale(1);
          }
        }
        
        .hooxi-message.user {
          align-self: flex-end;
        }
        
        .hooxi-message.assistant {
          align-self: flex-start;
        }
        
        .hooxi-message-bubble {
          padding: 14px 18px;
          font-size: 15px;
          line-height: 1.6;
          position: relative;
          white-space: pre-wrap;
          word-wrap: break-word;
        }
        
        .hooxi-message.user .hooxi-message-bubble {
          background: linear-gradient(135deg, #0D9488 0%, #14B8A6 100%);
          color: white;
          border-radius: 18px 18px 4px 18px;
          box-shadow: 
            0 4px 16px rgba(13, 148, 136, 0.25),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
          font-weight: 500;
        }
        
        .hooxi-message.assistant .hooxi-message-bubble {
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(10px);
          color: #1a1a1a;
          border: 1.5px solid rgba(13, 148, 136, 0.1);
          border-radius: 18px 18px 18px 4px;
          box-shadow: 
            0 4px 16px rgba(13, 148, 136, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        }
        
        .hooxi-message-time {
          font-size: 11px;
          color: #97A1AF;
          margin-top: 6px;
          padding: 0 6px;
          font-weight: 600;
          letter-spacing: 0.02em;
        }
        
        .hooxi-message.user .hooxi-message-time {
          text-align: right;
        }
        
        /* 타이핑 인디케이터 - 프리미엄 */
        .hooxi-typing {
          display: flex;
          gap: 5px;
          padding: 14px 18px;
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(10px);
          border: 1.5px solid rgba(13, 148, 136, 0.1);
          border-radius: 18px 18px 18px 4px;
          width: fit-content;
          box-shadow: 
            0 4px 16px rgba(13, 148, 136, 0.08),
            inset 0 1px 0 rgba(255, 255, 255, 0.8);
        }
        
        .hooxi-typing-dot {
          width: 9px;
          height: 9px;
          border-radius: 50%;
          background: linear-gradient(135deg, #0D9488 0%, #14B8A6 100%);
          animation: typingBounce 1.4s infinite ease-in-out;
        }
        
        .hooxi-typing-dot:nth-child(2) {
          animation-delay: 0.2s;
        }
        
        .hooxi-typing-dot:nth-child(3) {
          animation-delay: 0.4s;
        }
        
        @keyframes typingBounce {
          0%, 60%, 100% { 
            transform: translateY(0); 
            opacity: 0.5;
          }
          30% { 
            transform: translateY(-8px); 
            opacity: 1;
          }
        }
        
        /* 빠른 답변 - 프리미엄 디자인 */
        .hooxi-quick-replies {
          display: flex;
          gap: 10px;
          padding: 0 28px 16px;
          overflow-x: auto;
          flex-wrap: wrap;
        }
        
        .hooxi-quick-replies::-webkit-scrollbar {
          height: 0;
        }
        
        .hooxi-quick-reply {
          padding: 10px 18px;
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(10px);
          border: 1.5px solid rgba(13, 148, 136, 0.2);
          border-radius: 20px;
          color: #0D9488;
          font-size: 13px;
          font-weight: 600;
          cursor: pointer;
          white-space: nowrap;
          transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
          flex-shrink: 0;
          display: flex;
          align-items: center;
          gap: 6px;
          box-shadow: 0 2px 8px rgba(13, 148, 136, 0.06);
        }
        
        .hooxi-quick-reply:hover {
          background: linear-gradient(135deg, #0D9488 0%, #14B8A6 100%);
          color: white;
          border-color: transparent;
          transform: translateY(-2px);
          box-shadow: 0 6px 20px rgba(13, 148, 136, 0.25);
        }
        
        .hooxi-quick-reply:active {
          transform: translateY(0);
        }
        
        /* 입력 영역 - 프리미엄 */
        .hooxi-input-area {
          padding: 20px 28px 24px;
          background: rgba(255, 255, 255, 0.95);
          backdrop-filter: blur(20px);
          border-top: 1px solid rgba(13, 148, 136, 0.08);
          display: flex;
          gap: 12px;
          align-items: center;
          flex-shrink: 0;
          box-shadow: 0 -4px 24px rgba(13, 148, 136, 0.04);
        }
        
        .hooxi-input {
          flex: 1;
          padding: 14px 18px;
          border: 1.5px solid rgba(13, 148, 136, 0.15);
          border-radius: 16px;
          font-family: inherit;
          font-size: 15px;
          color: #1a1a1a;
          outline: none;
          transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
          resize: none;
          max-height: 120px;
          background: rgba(255, 255, 255, 0.8);
          backdrop-filter: blur(10px);
          box-shadow: inset 0 1px 2px rgba(0, 0, 0, 0.04);
        }
        
        .hooxi-input:focus {
          border-color: #0D9488;
          box-shadow: 
            0 0 0 4px rgba(13, 148, 136, 0.08),
            inset 0 1px 2px rgba(0, 0, 0, 0.04);
          background: white;
        }
        
        .hooxi-input::placeholder {
          color: #97A1AF;
          font-weight: 500;
        }
        
        .hooxi-send-button {
          width: 52px;
          height: 52px;
          border: none;
          background: linear-gradient(135deg, #0D9488 0%, #14B8A6 100%);
          border-radius: 16px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          transition: all 0.25s cubic-bezier(0.34, 1.56, 0.64, 1);
          flex-shrink: 0;
          box-shadow: 
            0 4px 16px rgba(13, 148, 136, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
          position: relative;
        }
        
        .hooxi-send-button::before {
          content: '';
          position: absolute;
          inset: 0;
          border-radius: 16px;
          background: linear-gradient(135deg, rgba(255, 255, 255, 0.2), transparent);
          opacity: 0;
          transition: opacity 0.25s ease;
        }
        
        .hooxi-send-button:hover {
          transform: translateY(-2px) scale(1.05);
          box-shadow: 
            0 8px 24px rgba(13, 148, 136, 0.4),
            inset 0 1px 0 rgba(255, 255, 255, 0.3);
        }
        
        .hooxi-send-button:hover::before {
          opacity: 1;
        }
        
        .hooxi-send-button:active {
          transform: translateY(0) scale(0.98);
        }
        
        .hooxi-send-button:disabled {
          opacity: 0.4;
          cursor: not-allowed;
          transform: none;
          box-shadow: 0 2px 8px rgba(13, 148, 136, 0.15);
        }
        
        .hooxi-send-button svg {
          width: 22px;
          height: 22px;
          stroke: white;
          stroke-width: 2.5;
          filter: drop-shadow(0 1px 2px rgba(0, 0, 0, 0.1));
        }
        
        /* 모바일 반응형 */
        @media (max-width: 480px) {
          .hooxi-float-button {
            width: 60px;
            height: 60px;
            right: 20px;
            bottom: 20px;
          }
          
          .hooxi-float-button svg {
            width: 28px;
            height: 28px;
          }
          
          .hooxi-chat-window.floating {
            width: calc(100vw - 24px);
            height: calc(100vh - 110px);
            right: 12px;
            bottom: 96px;
            border-radius: 20px;
          }
          
          .hooxi-chat-window.sidebar {
            width: 100vw;
            height: 100vh;
            border-radius: 0;
          }
          
          .hooxi-header {
            padding: 20px 24px;
          }
          
          .hooxi-messages {
            padding: 20px;
          }
          
          .hooxi-input-area {
            padding: 16px 20px 20px;
          }
          
          .hooxi-welcome {
            padding: 36px 20px 20px;
          }
          
          .hooxi-welcome-icon {
            width: 76px;
            height: 76px;
            font-size: 38px;
          }
        }
        
        /* 애니메이션 감소 선호 */
        @media (prefers-reduced-motion: reduce) {
          *, *::before, *::after {
            animation-duration: 0.01ms !important;
            transition-duration: 0.01ms !important;
          }
        }
      `}</style>

      {/* 플로팅 버튼 */}
      <button 
        className={`hooxi-float-button ${isOpen ? 'open' : ''}`}
        onClick={() => setIsOpen(!isOpen)}
        aria-label={isOpen ? '채팅 닫기' : '채팅 열기'}
      >
        {isOpen ? (
          <svg viewBox="0 0 24 24" fill="none">
            <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        ) : (
          <svg viewBox="0 0 24 24" fill="none">
            <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" strokeLinecap="round" strokeLinejoin="round"/>
          </svg>
        )}
      </button>

      {/* 채팅 윈도우 */}
      <div className={`hooxi-chat-window ${isSidebarMode ? 'sidebar' : 'floating'} ${isOpen ? 'open' : ''}`}>
        {/* 헤더 */}
        <div className="hooxi-header">
          <div className="hooxi-header-left">
            <div className="hooxi-header-icon">🌱</div>
            <div className="hooxi-header-title-wrap">
              <div className="hooxi-header-title">후시봇</div>
              <div className="hooxi-header-subtitle">AI 탄소배출 전문 상담</div>
            </div>
          </div>
          <div className="hooxi-header-buttons">
            <button 
              className="hooxi-header-button"
              data-tooltip="새 AI 채팅하기"
              onClick={handleNewChat}
              aria-label="새 AI 채팅하기"
            >
              <svg viewBox="0 0 24 24" fill="none">
                <path d="M12 5v14M5 12h14" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
            <button 
              className="hooxi-header-button"
              data-tooltip="채팅 모드 전환"
              onClick={() => setIsSidebarMode(!isSidebarMode)}
              aria-label="채팅 모드 전환"
            >
              <svg viewBox="0 0 24 24" fill="none">
                <path d="M3 9h18M3 15h18M9 3v18" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
            <button 
              className="hooxi-header-button"
              data-tooltip="챗봇 닫기"
              onClick={() => setIsOpen(false)}
              aria-label="챗봇 닫기"
            >
              <svg viewBox="0 0 24 24" fill="none">
                <path d="M18 6L6 18M6 6l12 12" strokeLinecap="round" strokeLinejoin="round"/>
              </svg>
            </button>
          </div>
        </div>

        {/* 메시지 영역 */}
        <div className="hooxi-messages">
          {showWelcome ? (
            <div className="hooxi-welcome">
              <div className="hooxi-welcome-icon-wrap">
                <div className="hooxi-welcome-icon">🌍</div>
              </div>
              <div>
                <h2>안녕하세요! 후시봇입니다</h2>
                <p>탄소배출 관리부터 배출권 거래까지<br/>전문적으로 안내해드립니다</p>
              </div>
              <div className="hooxi-welcome-options">
                {welcomeOptions.map((option, index) => (
                  <button
                    key={index}
                    className="hooxi-welcome-option"
                    onClick={() => handleWelcomeOption(option)}
                  >
                    <div className="hooxi-welcome-option-icon">{option.icon}</div>
                    <div className="hooxi-welcome-option-content">
                      <div className="hooxi-welcome-option-title">{option.text}</div>
                      <div className="hooxi-welcome-option-desc">{option.desc}</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <>
              {messages.map((message) => (
                <div key={message.id} className={`hooxi-message ${message.role}`}>
                  <div className="hooxi-message-bubble">{message.text}</div>
                  <div className="hooxi-message-time">
                    {new Date(message.timestamp).toLocaleTimeString('ko-KR', {
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </div>
                </div>
              ))}
              {isTyping && (
                <div className="hooxi-typing">
                  <div className="hooxi-typing-dot"></div>
                  <div className="hooxi-typing-dot"></div>
                  <div className="hooxi-typing-dot"></div>
                </div>
              )}
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* 빠른 답변 */}
        {!showWelcome && messages.length > 0 && (
          <div className="hooxi-quick-replies">
            {quickReplies.map((reply, index) => (
              <button
                key={index}
                className="hooxi-quick-reply"
                onClick={() => handleQuickReply(reply)}
              >
                <span>{reply.icon}</span>
                <span>{reply.text}</span>
              </button>
            ))}
          </div>
        )}

        {/* 입력 영역 */}
        <div className="hooxi-input-area">
          <input
            type="text"
            className="hooxi-input"
            placeholder="메시지를 입력하세요..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            aria-label="메시지 입력"
          />
          <button
            className="hooxi-send-button"
            onClick={() => handleSendMessage()}
            disabled={!inputValue.trim()}
            aria-label="메시지 전송"
          >
            <svg viewBox="0 0 24 24" fill="none">
              <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>
      </div>
    </>
  );
};

export default HooxiBotPro;