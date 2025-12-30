/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{ts,tsx,js,jsx}",
    "./agent/**/*.{ts,tsx,js,jsx}",
  ],
  theme: {
    extend: {
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
        // 후시파트너스 커스텀 라운드
        hooxi: "0.875rem", // 14px
        "hooxi-sm": "0.5rem", // 8px
        "hooxi-md": "0.75rem", // 12px
        "hooxi-lg": "1.25rem", // 20px
        "hooxi-xl": "1.5rem", // 24px
      },
      fontFamily: {
        // Pretendard 폰트 추가
        pretendard: ['Pretendard', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'sans-serif'],
        sans: ['Pretendard', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif'],
      },
      fontSize: {
        // 후시파트너스 타이포그래피
        'xs-hooxi': '0.75rem',      // 12px
        'sm-hooxi': '0.8125rem',    // 13px
        'base-hooxi': '0.9375rem',  // 15px
        'lg-hooxi': '1.0625rem',    // 17px
        'xl-hooxi': '1.25rem',      // 20px
        '2xl-hooxi': '1.5rem',      // 24px
      },
      components: {
        ".scrollbar-pretty":
          "overflow-y-scroll [&::-webkit-scrollbar]:w-1.5 [&::-webkit-scrollbar-thumb]:rounded-full [&::-webkit-scrollbar-thumb]:bg-gray-300 [&::-webkit-scrollbar-track]:bg-transparent",
      },
      colors: {
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        chart: {
          1: "hsl(var(--chart-1))",
          2: "hsl(var(--chart-2))",
          3: "hsl(var(--chart-3))",
          4: "hsl(var(--chart-4))",
          5: "hsl(var(--chart-5))",
        },
        // 후시파트너스 브랜드 컬러 추가
        hooxi: {
          primary: "#0D9488",
          "primary-dark": "#0F766E",
          "primary-light": "#E0F2F1",
          "primary-hover": "#14B8A6",
          secondary: "#97A1AF",
          "secondary-dark": "#6B7280",
          "secondary-light": "#F1F5F9",
          black: "#000000",
          "black-soft": "#1A1A1A",
          bg: "#FAFAFA",
          surface: "#FFFFFF",
          border: "#E5E7EB",
          "border-light": "#F3F4F6",
          success: "#10B981",
          error: "#EF4444",
          warning: "#F59E0B",
          info: "#3B82F6",
        },
      },
      boxShadow: {
        // 후시파트너스 커스텀 그림자 (Teal 틴트)
        "hooxi-sm": "0 2px 8px rgba(13, 148, 136, 0.08)",
        "hooxi-md": "0 8px 24px rgba(13, 148, 136, 0.12)",
        "hooxi-lg": "0 16px 48px rgba(13, 148, 136, 0.16)",
        "hooxi-xl": "0 24px 64px rgba(13, 148, 136, 0.20)",
        "hooxi-black": "0 4px 12px rgba(0, 0, 0, 0.15)",
      },
      backgroundImage: {
        // 후시파트너스 그라디언트
        "gradient-hooxi": "linear-gradient(135deg, #0D9488 0%, #14B8A6 100%)",
        "gradient-hooxi-dark": "linear-gradient(135deg, #0F766E 0%, #0D9488 100%)",
        "gradient-hooxi-subtle": "linear-gradient(180deg, #FAFAFA 0%, #E0F2F1 100%)",
        "gradient-hooxi-secondary": "linear-gradient(135deg, #97A1AF 0%, #CBD5E1 100%)",
      },
      animation: {
        // 후시파트너스 커스텀 애니메이션
        "float": "float 3s ease-in-out infinite",
        "gentle-bounce": "gentle-bounce 2s ease-in-out infinite",
        "pulse-badge": "pulse-badge 2s ease-in-out infinite",
        "message-slide-in": "messageSlideIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-8px)" },
        },
        "gentle-bounce": {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%": { transform: "translateY(-3px)" },
        },
        "pulse-badge": {
          "0%, 100%": { transform: "scale(1)", opacity: "1" },
          "50%": { transform: "scale(1.2)", opacity: "0.8" },
        },
        messageSlideIn: {
          from: { 
            opacity: "0", 
            transform: "translateY(20px) scale(0.95)" 
          },
          to: { 
            opacity: "1", 
            transform: "translateY(0) scale(1)" 
          },
        },
      },
      transitionTimingFunction: {
        // 후시파트너스 이징 함수
        "hooxi-bounce": "cubic-bezier(0.34, 1.56, 0.64, 1)",
      },
    },
  },
  plugins: [
    require("tailwindcss-animate"), 
    require("tailwind-scrollbar"),
  ],
};