import { useState, useMemo } from "react";
import { ChevronLeft, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import { motion, AnimatePresence } from "framer-motion";

interface ChatOpenersProps {
  chatOpeners: string[];
  onSelectOpener: (opener: string) => void;
  disabled: boolean;
}

// Hooxi-specific opener icons mapping
const getOpenerIcon = (opener: string): string => {
  if (opener.includes("측정") || opener.includes("배출량")) return "📊";
  if (opener.includes("판매")) return "💰";
  if (opener.includes("구매")) return "🛒";
  if (opener.includes("상담") || opener.includes("연결")) return "👤";
  return "💬";
};

export function ChatOpeners({ chatOpeners, onSelectOpener, disabled }: ChatOpenersProps) {
  const [currentPage, setCurrentPage] = useState(0);
  const itemsPerPage = 4;
  const totalPages = Math.ceil(chatOpeners.length / itemsPerPage);
  const shouldShowCarousel = chatOpeners.length > itemsPerPage;

  const currentItems = useMemo(() => {
    const startIndex = currentPage * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    return chatOpeners.slice(startIndex, endIndex);
  }, [currentPage, chatOpeners, itemsPerPage]);

  const goToNextPage = () => {
    setCurrentPage((prev) => (prev + 1) % totalPages);
  };

  const goToPrevPage = () => {
    setCurrentPage((prev) => (prev - 1 + totalPages) % totalPages);
  };

  const openerButtonHandler = (opener: string) => () => {
    if (disabled) {
      return;
    }
    onSelectOpener(opener);
  };

  return (
    <div className="flex flex-col gap-4 w-full max-w-3xl mx-auto px-4">
      <div className="relative">
        <AnimatePresence mode="wait">
          <motion.div
            key={currentPage}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            transition={{ duration: 0.3, ease: "easeOut" }}
            className="grid grid-cols-1 sm:grid-cols-2 gap-4"
          >
            {currentItems.map((opener, index) => {
              const icon = getOpenerIcon(opener);
              return (
                <button
                  key={`${currentPage}-${index}`}
                  onClick={openerButtonHandler(opener)}
                  disabled={disabled}
                  className={cn(
                    "group relative overflow-hidden rounded-2xl border-2 border-hooxi-primary/20 bg-white hover:bg-hooxi-primary-light/30 hover:border-hooxi-primary/50 transition-all duration-300 p-5 text-left shadow-hooxi-sm hover:shadow-hooxi-md min-h-[6rem] flex items-center gap-4 cursor-pointer",
                    disabled && "opacity-50 cursor-not-allowed hover:bg-white hover:border-hooxi-primary/20 hover:shadow-hooxi-sm"
                  )}
                >
                  <div className="flex-shrink-0 text-3xl transition-transform duration-300 group-hover:scale-110">
                    {icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-[15px] font-medium text-gray-800 group-hover:text-hooxi-primary transition-colors duration-300 break-keep leading-relaxed">
                      {opener}
                    </p>
                  </div>
                  <div className="absolute inset-0 bg-gradient-to-r from-hooxi-primary/0 via-hooxi-primary/5 to-hooxi-primary/0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 pointer-events-none" />
                </button>
              );
            })}
          </motion.div>
        </AnimatePresence>
      </div>

      {shouldShowCarousel && (
        <div className="flex items-center justify-center gap-3 mt-2">
          <button
            onClick={goToPrevPage}
            className="flex h-9 w-9 items-center justify-center rounded-full border-2 border-hooxi-primary/30 bg-white hover:bg-hooxi-primary-light/30 hover:border-hooxi-primary transition-all duration-200"
            aria-label="Previous page"
          >
            <ChevronLeft className="h-4 w-4 text-hooxi-primary" />
          </button>

          <div className="flex items-center gap-2">
            {Array.from({ length: totalPages }).map((_, index) => (
              <button
                key={index}
                onClick={() => setCurrentPage(index)}
                className={cn(
                  "h-2.5 rounded-full transition-all duration-300",
                  index === currentPage
                    ? "w-8 bg-hooxi-primary shadow-[0_0_8px_rgba(13,148,136,0.4)]"
                    : "w-2.5 bg-gray-300 hover:bg-gray-400"
                )}
                aria-label={`Go to page ${index + 1}`}
              />
            ))}
          </div>

          <button
            onClick={goToNextPage}
            className="flex h-9 w-9 items-center justify-center rounded-full border-2 border-hooxi-primary/30 bg-white hover:bg-hooxi-primary-light/30 hover:border-hooxi-primary transition-all duration-200"
            aria-label="Next page"
          >
            <ChevronRight className="h-4 w-4 text-hooxi-primary" />
          </button>
        </div>
      )}
    </div>
  );
}
