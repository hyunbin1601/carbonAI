import { useState, useEffect, useRef } from "react";

interface UseTypingEffectOptions {
  /** 타이핑 속도 (ms per character) */
  speed?: number;
  /** 타이핑 효과 활성화 여부 */
  enabled?: boolean;
  /** 완료 콜백 */
  onComplete?: () => void;
}

/**
 * ChatGPT 스타일의 타이핑 효과를 제공하는 훅
 * 텍스트를 점진적으로 표시합니다.
 */
export function useTypingEffect(
  text: string,
  options: UseTypingEffectOptions = {}
) {
  const { speed = 15, enabled = true, onComplete } = options;

  const [displayedText, setDisplayedText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const previousTextRef = useRef("");
  const indexRef = useRef(0);
  const animationRef = useRef<number | null>(null);

  useEffect(() => {
    // 타이핑 효과가 비활성화되면 전체 텍스트를 바로 표시
    if (!enabled) {
      setDisplayedText(text);
      setIsTyping(false);
      return;
    }

    // 텍스트가 비어있으면 초기화
    if (!text) {
      setDisplayedText("");
      setIsTyping(false);
      previousTextRef.current = "";
      indexRef.current = 0;
      return;
    }

    // 텍스트가 이전과 같으면 아무것도 하지 않음
    if (text === previousTextRef.current) {
      return;
    }

    // 텍스트가 이전 텍스트로 시작하면 (추가된 경우) 이어서 타이핑
    // 그렇지 않으면 처음부터 시작
    if (text.startsWith(previousTextRef.current) && previousTextRef.current.length > 0) {
      // 이전 텍스트에서 이어서 타이핑
      indexRef.current = previousTextRef.current.length;
    } else {
      // 완전히 새로운 텍스트 - 처음부터 시작
      indexRef.current = 0;
      setDisplayedText("");
    }

    previousTextRef.current = text;
    setIsTyping(true);

    // 이전 애니메이션 취소
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current);
    }

    let lastTime = performance.now();

    const animate = (currentTime: number) => {
      const elapsed = currentTime - lastTime;

      if (elapsed >= speed) {
        const charsToAdd = Math.floor(elapsed / speed);
        const newIndex = Math.min(indexRef.current + charsToAdd, text.length);

        if (newIndex > indexRef.current) {
          indexRef.current = newIndex;
          setDisplayedText(text.slice(0, newIndex));
          lastTime = currentTime;
        }

        if (indexRef.current >= text.length) {
          setIsTyping(false);
          onComplete?.();
          return;
        }
      }

      animationRef.current = requestAnimationFrame(animate);
    };

    animationRef.current = requestAnimationFrame(animate);

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [text, speed, enabled, onComplete]);

  // 타이핑 완료 시 또는 비활성화 시 전체 텍스트 반환
  const finalText = enabled ? displayedText : text;

  return {
    displayedText: finalText,
    isTyping,
    isComplete: finalText === text,
  };
}
