"use client";

import { useEffect, useState, useMemo, useRef, useCallback, memo } from "react";
import Map from "react-map-gl/maplibre";
import DeckGL from "@deck.gl/react";
import { ScatterplotLayer, PathLayer, PolygonLayer, GeoJsonLayer } from "@deck.gl/layers";
import { HexagonLayer } from "@deck.gl/aggregation-layers";
import type { Layer, PickingInfo } from "@deck.gl/core";
import { cn } from "@/lib/utils";
import "maplibre-gl/dist/maplibre-gl.css";

// 전역 활성 맵 관리 (WebGL 컨텍스트 제한 준수)
const activeMapInstances = new Set<string>();
const MAX_ACTIVE_MAPS = 1; // 최대 1개 맵만 동시 활성화 (각 맵은 2개 WebGL context 사용)

const registerMap = (id: string): boolean => {
  if (activeMapInstances.size >= MAX_ACTIVE_MAPS && !activeMapInstances.has(id)) {
    return false; // 제한 초과
  }
  activeMapInstances.add(id);
  return true;
};

const unregisterMap = (id: string) => {
  activeMapInstances.delete(id);
};

interface MapConfig {
  initialViewState?: {
    longitude?: number;
    latitude?: number;
    zoom?: number;
    pitch?: number;
    bearing?: number;
  };
  layers?: Array<{
    type: "scatterplot" | "path" | "polygon" | "hexagon" | "geojson";
    data: any;
    [key: string]: any;
  }>;
  style?: string; // Map style URL
  tooltip?: boolean;
}

interface MapRendererProps {
  config: string | MapConfig;
  className?: string;
}

// 기본은 서울 중심 좌표
const DEFAULT_VIEW_STATE = {
  longitude: 126.9780,
  latitude: 37.5665,
  zoom: 11,
  pitch: 0,
  bearing: 0,
};

// OSM 타일 서버 (다크 모드 지원)
const LIGHT_MAP_STYLE = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json";
const DARK_MAP_STYLE = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

// 개선된 색상 팔레트 (라이트/다크 모드 최적화)
const COLOR_PALETTES = {
  light: {
    scatterplot: [59, 130, 246, 200],      // Blue
    path: [239, 68, 68, 200],              // Red
    polygon: [34, 197, 94, 150],           // Green
    hexagon: [168, 85, 247, 180],          // Purple
    geojson: [249, 115, 22, 180],          // Orange
  },
  dark: {
    scatterplot: [96, 165, 250, 220],      // Light Blue
    path: [248, 113, 113, 220],            // Light Red
    polygon: [74, 222, 128, 170],          // Light Green
    hexagon: [196, 181, 253, 200],         // Light Purple
    geojson: [251, 146, 60, 200],          // Light Orange
  },
};

export function MapRenderer({ config, className }: MapRendererProps) {
  const [mapConfig, setMapConfig] = useState<MapConfig | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [hoveredObject, setHoveredObject] = useState<any>(null);
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [isMounted, setIsMounted] = useState(false);
  const [mapLoaded, setMapLoaded] = useState(false);
  const [dataSamplingInfo, setDataSamplingInfo] = useState<{
    original: number;
    sampled: number;
  } | null>(null);
  const [layersVisible, setLayersVisible] = useState<boolean[]>([]);
  const [showLegend, setShowLegend] = useState(true);
  const [viewState, setViewState] = useState(DEFAULT_VIEW_STATE);
  const [isInView, setIsInView] = useState(false);
  const [canActivate, setCanActivate] = useState(false);
  const [isReady, setIsReady] = useState(false); // 초기 로드 완료 플래그

  // DeckGL 및 Map 인스턴스 ref
  const deckRef = useRef<any>(null);
  const mapRef = useRef<any>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // 컴포넌트 고유 ID (재렌더링 시에도 유지)
  const instanceId = useRef(`map-${Math.random().toString(36).substr(2, 9)}`).current;

  // Intersection Observer + 전역 맵 카운터로 WebGL 컨텍스트 최적화
  useEffect(() => {
    if (!containerRef.current) return;

    let retryInterval: NodeJS.Timeout | null = null;
    let retryTimeout: NodeJS.Timeout | null = null;
    let initialCheckDone = false;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const isVisible = entry.isIntersecting;

          if (isVisible) {
            // 뷰포트에 들어올 때만 상태 업데이트
            setIsInView(true);

            // 초기 체크 시 약간의 지연 추가 (동시 로드 방지)
            const activationDelay = initialCheckDone ? 0 : Math.random() * 300;

            setTimeout(() => {
              // 활성화 시도
              const canRegister = registerMap(instanceId);

              if (canRegister) {
                // 등록 성공 - 다음 프레임에서 렌더링 시작 (WebGL context 생성 지연)
                setCanActivate(true);
                requestAnimationFrame(() => {
                  setIsReady(true);
                });
              } else {
                // 등록 실패 - placeholder 표시
                setCanActivate(false);
                setIsReady(true);

                // 재시도 (다른 맵이 언마운트되면)
                retryInterval = setInterval(() => {
                  const retry = registerMap(instanceId);
                  if (retry) {
                    setCanActivate(true);
                    requestAnimationFrame(() => {
                      setIsReady(true);
                    });
                    if (retryInterval) clearInterval(retryInterval);
                  }
                }, 500);

                // 5초 후 포기
                retryTimeout = setTimeout(() => {
                  if (retryInterval) clearInterval(retryInterval);
                }, 5000);
              }
            }, activationDelay);

            initialCheckDone = true;
          } else {
            // 뷰포트에서 벗어날 때 즉시 정리
            setIsInView(false);
            unregisterMap(instanceId);
            setCanActivate(false);
            setIsReady(true); // 체크는 완료됨

            // 재시도 타이머 정리
            if (retryInterval) clearInterval(retryInterval);
            if (retryTimeout) clearTimeout(retryTimeout);
          }
        });
      },
      {
        rootMargin: '50px',
        threshold: 0.1,
      }
    );

    // 초기 체크를 위해 약간 지연 후 observe 시작
    const startDelay = setTimeout(() => {
      if (containerRef.current) {
        observer.observe(containerRef.current);
      }
    }, 50);

    return () => {
      clearTimeout(startDelay);
      observer.disconnect();
      unregisterMap(instanceId);
      if (retryInterval) clearInterval(retryInterval);
      if (retryTimeout) clearTimeout(retryTimeout);
    };
  }, [instanceId]);

  // 클라이언트 마운트 체크
  useEffect(() => {
    setIsMounted(true);

    // cleanup: 컴포넌트 언마운트 시 WebGL 리소스 정리
    return () => {
      setMapLoaded(false);

      // Map 리소스 정리 - ref를 지역 변수로 복사
      const currentMapRef = mapRef.current;
      const currentDeckRef = deckRef.current;

      if (currentMapRef) {
        try {
          const mapInstance = currentMapRef.getMap();
          if (mapInstance) {
            mapInstance.remove();
          }
        } catch (e) {
          console.warn('Map cleanup warning:', e);
        }
      }

      // DeckGL 리소스 정리
      if (currentDeckRef) {
        try {
          currentDeckRef.finalize();
        } catch (e) {
          console.warn('DeckGL cleanup warning:', e);
        }
      }
    };
  }, []);

  // 다크 모드 감지
  useEffect(() => {
    const checkDarkMode = () => {   // dark mode 함수
      const isDark = document.documentElement.classList.contains("dark");
      setIsDarkMode(isDark);
    };

    checkDarkMode();
    const observer = new MutationObserver(checkDarkMode);
    observer.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });

    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    setIsLoading(true);
    setError(null);
    setMapConfig(null);
    setMapLoaded(false); // 새 config 로딩 시 맵 로드 상태 초기화

    try {
      let parsedConfig: MapConfig;

      if (typeof config === "string") {
        parsedConfig = JSON.parse(config);
      } else {
        parsedConfig = config;
      }

      setMapConfig(parsedConfig);
      // 레이어 가시성 초기화 (모두 보이도록)
      if (parsedConfig.layers) {
        setLayersVisible(new Array(parsedConfig.layers.length).fill(true));
      }
      // 초기 뷰 상태 설정
      setViewState({
        ...DEFAULT_VIEW_STATE,
        ...parsedConfig.initialViewState,
      });
      setIsLoading(false);
    } catch (err) {
      console.error("Map config parsing error:", err);
      setError(err instanceof Error ? err.message : "맵을 렌더링할 수 없습니다.");
      setIsLoading(false);
    }
  }, [config]);

  // 데이터 샘플링 (너무 많은 데이터포인트는 성능 저하)
  const sampleData = (data: any[], maxPoints: number = 5000) => {
    if (!Array.isArray(data) || data.length <= maxPoints) return data;

    const step = Math.ceil(data.length / maxPoints);
    return data.filter((_, i) => i % step === 0);
  };

  // deck.gl 레이어 생성
  const layers: Layer[] = useMemo(() => {
    if (!mapConfig?.layers) return [];

    let totalOriginal = 0;
    let totalSampled = 0;

    const palette = isDarkMode ? COLOR_PALETTES.dark : COLOR_PALETTES.light;

    const createdLayers = mapConfig.layers.map((layerConfig, index) => {
      // 레이어가 숨겨진 경우 스킵
      if (!layersVisible[index]) return null;

      // type과 data를 제외한 나머지 속성만 추출
      const { type, data, ...otherProps } = layerConfig;

      // 데이터 카운트
      const originalCount = Array.isArray(data) ? data.length : 0;
      totalOriginal += originalCount;

      // 데이터 샘플링 (대용량 데이터 처리)
      const sampledData = sampleData(data);
      const sampledCount = Array.isArray(sampledData) ? sampledData.length : 0;
      totalSampled += sampledCount;

      const commonProps = {
        id: `layer-${index}`,
        data: sampledData,
        pickable: true,
        ...otherProps,
      };

      switch (type) {
        case "scatterplot":
          return new ScatterplotLayer({
            ...commonProps,
            getPosition: (d: any) => d.position || [d.longitude, d.latitude],
            getRadius: (d: any) => d.radius || 100,
            getFillColor: (d: any) => d.color || palette.scatterplot,
            radiusMinPixels: layerConfig.radiusMinPixels || 5,
            radiusMaxPixels: layerConfig.radiusMaxPixels || 30,
          });

        case "path":
          return new PathLayer({
            ...commonProps,
            getPath: (d: any) => d.path,
            getColor: (d: any) => d.color || palette.path,
            getWidth: (d: any) => d.width || 5,
            widthMinPixels: layerConfig.widthMinPixels || 2,
          });

        case "polygon":
          return new PolygonLayer({
            ...commonProps,
            getPolygon: (d: any) => d.polygon,
            getFillColor: (d: any) => d.fillColor || palette.polygon,
            getLineColor: (d: any) => d.lineColor || (isDarkMode ? [255, 255, 255, 100] : [0, 0, 0, 100]),
            getLineWidth: layerConfig.lineWidth || 1,
            filled: true,
            extruded: layerConfig.extruded || false,
            getElevation: (d: any) => d.elevation || 0,
          });

        case "hexagon":
          return new HexagonLayer({
            ...commonProps,
            getPosition: (d: any) => d.position || [d.longitude, d.latitude],
            getColorWeight: (d: any) => d.weight || 1,
            getElevationWeight: (d: any) => d.weight || 1,
            colorAggregation: 'SUM',
            elevationAggregation: 'SUM',
            elevationScale: layerConfig.elevationScale || 10,
            radius: layerConfig.radius || 500,
            coverage: layerConfig.coverage || 0.8,
            extruded: layerConfig.extruded !== undefined ? layerConfig.extruded : true,
          });

        case "geojson":
          return new GeoJsonLayer({
            ...commonProps,
            filled: true,
            stroked: true,
            getFillColor: (d: any) => d.properties?.fillColor || palette.geojson,
            getLineColor: (d: any) => d.properties?.lineColor || (isDarkMode ? [255, 255, 255, 100] : [0, 0, 0, 100]),
            getLineWidth: layerConfig.lineWidth || 1,
            pointRadiusMinPixels: 3,
            pointRadiusMaxPixels: 10,
          });

        default:
          return null;
      }
    }).filter(Boolean) as Layer[];

    // 샘플링 정보 업데이트
    if (totalOriginal > totalSampled) {
      setDataSamplingInfo({ original: totalOriginal, sampled: totalSampled });
    } else {
      setDataSamplingInfo(null);
    }

    return createdLayers;
  }, [mapConfig, isDarkMode, layersVisible]);

  // 툴팁 정보 - useCallback으로 메모이제이션
  const handleHover = useCallback((info: PickingInfo) => {
    setHoveredObject(info.object);
  }, []);

  // viewState 업데이트 throttle (드래그 시 과도한 리렌더링 방지)
  const lastUpdateTime = useRef(0);
  const handleViewStateChange = useCallback(({ viewState: newViewState }: any) => {
    const now = Date.now();
    // 16ms (60fps) throttle
    if (now - lastUpdateTime.current > 16) {
      setViewState(newViewState);
      lastUpdateTime.current = now;
    }
  }, []);

  // mapStyle을 메모이제이션하여 불필요한 재로드 방지
  const mapStyle = useMemo(() =>
    mapConfig?.style || (isDarkMode ? DARK_MAP_STYLE : LIGHT_MAP_STYLE),
    [mapConfig?.style, isDarkMode]
  );

  // 컨트롤 함수들
  const handleZoomIn = useCallback(() => {
    setViewState(prev => ({ ...prev, zoom: Math.min(prev.zoom + 1, 20) }));
  }, []);

  const handleZoomOut = useCallback(() => {
    setViewState(prev => ({ ...prev, zoom: Math.max(prev.zoom - 1, 0) }));
  }, []);

  const handleResetView = useCallback(() => {
    setViewState({
      ...DEFAULT_VIEW_STATE,
      ...mapConfig?.initialViewState,
    });
  }, [mapConfig?.initialViewState]);

  const toggleLayer = useCallback((index: number) => {
    setLayersVisible(prev => {
      const newVisible = [...prev];
      newVisible[index] = !newVisible[index];
      return newVisible;
    });
  }, []);

  // 툴팁 데이터 포맷팅
  const formatTooltipData = useCallback((obj: any) => {
    if (!obj) return null;

    const entries = Object.entries(obj).filter(([key]) =>
      !key.startsWith('_') && key !== 'index'
    );

    if (entries.length === 0) return null;

    return entries.slice(0, 8); // 최대 8개 항목만 표시
  }, []);

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

  if (!isMounted || isLoading || !mapConfig) {
    return (
      <div
        ref={containerRef}
        className={cn(
          "rounded-xl bg-muted/50 dark:bg-zinc-900 p-4 border border-border/30 dark:border-zinc-700",
          className
        )}
      >
        <div className="text-center text-sm text-muted-foreground py-4">
          맵 렌더링 중...
        </div>
      </div>
    );
  }

  // 초기 체크 완료 전까지는 무조건 placeholder (동시 로드 방지)
  if (!isReady) {
    return (
      <div
        ref={containerRef}
        className={cn(
          "rounded-xl bg-muted/50 dark:bg-zinc-900 p-4 border border-border/30 dark:border-zinc-700",
          className
        )}
      >
        <div className="relative w-full h-[500px] flex items-center justify-center">
          <div className="text-center">
            <div className="text-sm text-muted-foreground mb-2">
              🗺️ 지도 준비 중...
            </div>
            <div className="text-xs text-muted-foreground/70">
              WebGL 리소스 최적화를 위해 순차적으로 로드됩니다
            </div>
          </div>
        </div>
      </div>
    );
  }

  // 뷰포트에 보이지 않거나 활성화 제한 초과 시 placeholder 표시
  if (!isInView || !canActivate) {
    const message = !isInView
      ? "🗺️ 지도 (스크롤하여 활성화)"
      : "⏳ 대기 중 (다른 지도 닫으면 활성화)";
    const subMessage = !isInView
      ? "WebGL 리소스 절약을 위해 뷰포트 내에서만 렌더링됩니다"
      : `최대 ${MAX_ACTIVE_MAPS}개 지도만 동시 활성화 가능 (현재: ${activeMapInstances.size}/${MAX_ACTIVE_MAPS})`;

    return (
      <div
        ref={containerRef}
        className={cn(
          "rounded-xl bg-muted/50 dark:bg-zinc-900 p-4 border border-border/30 dark:border-zinc-700",
          className
        )}
      >
        <div className="relative w-full h-[500px] flex items-center justify-center">
          <div className="text-center">
            <div className="text-sm text-muted-foreground mb-2">
              {message}
            </div>
            <div className="text-xs text-muted-foreground/70">
              {subMessage}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // WebGL 지원 체크
  if (typeof window !== 'undefined') {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl') || canvas.getContext('webgl2');
    if (!gl) {
      return (
        <div className={cn("rounded-xl bg-muted/50 dark:bg-zinc-900 p-4 border border-red-500/30", className)}>
          <p className="text-sm text-red-500">WebGL을 지원하지 않는 브라우저입니다.</p>
        </div>
      );
    }
  }

  return (
    <div
      ref={containerRef}
      className={cn(
        "map-container rounded-xl overflow-hidden border border-border/50 dark:border-zinc-700/50 shadow-sm",
        className
      )}
    >
      <div className="relative w-full h-[500px]">
        <DeckGL
          ref={deckRef}
          viewState={viewState}
          onViewStateChange={handleViewStateChange}
          controller={true}
          layers={mapLoaded ? layers : []}
          onHover={handleHover}
          getTooltip={() => null}
          useDevicePixels={1}
          _typedArrayManagerProps={{
            overAlloc: 1,
            poolSize: 0
          }}
        >
          <Map
            ref={mapRef}
            reuseMaps={true}
            mapStyle={mapStyle}
            onLoad={() => {
              setTimeout(() => setMapLoaded(true), 100);
            }}
            onError={(e) => {
              console.warn('Map error:', e);
            }}
          />
        </DeckGL>

        {/* 개선된 툴팁 */}
        {mapConfig.tooltip !== false && hoveredObject && (() => {
          const tooltipData = formatTooltipData(hoveredObject);
          return tooltipData && (
            <div
              className="absolute bottom-4 left-4 bg-white/95 dark:bg-zinc-900/95 backdrop-blur-sm p-4 rounded-xl shadow-2xl border border-border/50 dark:border-zinc-700/50 max-w-sm animate-in fade-in slide-in-from-bottom-2 duration-200"
              style={{ pointerEvents: "none" }}
            >
              <div className="space-y-2">
                {tooltipData.map(([key, value], idx) => (
                  <div key={idx} className="flex justify-between gap-4 text-sm">
                    <span className="font-medium text-muted-foreground capitalize">
                      {key.replace(/([A-Z])/g, ' $1').trim()}:
                    </span>
                    <span className="text-foreground font-semibold text-right">
                      {typeof value === 'number' ? value.toLocaleString() : String(value)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          );
        })()}

        {/* 줌 컨트롤 */}
        <div className="absolute top-4 right-4 flex flex-col gap-2">
          <button
            onClick={handleZoomIn}
            className="bg-white dark:bg-zinc-800 hover:bg-gray-100 dark:hover:bg-zinc-700 p-2 rounded-lg shadow-lg border border-border/50 dark:border-zinc-700/50 transition-colors"
            title="Zoom In"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
            </svg>
          </button>
          <button
            onClick={handleZoomOut}
            className="bg-white dark:bg-zinc-800 hover:bg-gray-100 dark:hover:bg-zinc-700 p-2 rounded-lg shadow-lg border border-border/50 dark:border-zinc-700/50 transition-colors"
            title="Zoom Out"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 12H4" />
            </svg>
          </button>
          <button
            onClick={handleResetView}
            className="bg-white dark:bg-zinc-800 hover:bg-gray-100 dark:hover:bg-zinc-700 p-2 rounded-lg shadow-lg border border-border/50 dark:border-zinc-700/50 transition-colors"
            title="Reset View"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />
            </svg>
          </button>
          <button
            onClick={() => setShowLegend(!showLegend)}
            className="bg-white dark:bg-zinc-800 hover:bg-gray-100 dark:hover:bg-zinc-700 p-2 rounded-lg shadow-lg border border-border/50 dark:border-zinc-700/50 transition-colors"
            title="Toggle Legend"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>

        {/* 레이어 범례 */}
        {showLegend && mapConfig.layers && mapConfig.layers.length > 0 && (
          <div className="absolute top-4 left-4 bg-white/95 dark:bg-zinc-900/95 backdrop-blur-sm p-4 rounded-xl shadow-lg border border-border/50 dark:border-zinc-700/50 max-w-xs animate-in fade-in slide-in-from-left-2 duration-200">
            <h3 className="text-sm font-semibold mb-3 text-foreground">Layers</h3>
            <div className="space-y-2">
              {mapConfig.layers.map((layer, index) => {
                const palette = isDarkMode ? COLOR_PALETTES.dark : COLOR_PALETTES.light;
                const color = palette[layer.type as keyof typeof palette] || [128, 128, 128, 200];
                const rgbaColor = `rgba(${color[0]}, ${color[1]}, ${color[2]}, ${color[3] / 255})`;

                return (
                  <label
                    key={index}
                    className="flex items-center gap-2 cursor-pointer hover:bg-gray-100 dark:hover:bg-zinc-800 p-2 rounded-lg transition-colors"
                  >
                    <input
                      type="checkbox"
                      checked={layersVisible[index]}
                      onChange={() => toggleLayer(index)}
                      className="rounded border-gray-300"
                    />
                    <div
                      className="w-4 h-4 rounded-sm border border-border/50"
                      style={{ backgroundColor: rgbaColor }}
                    />
                    <span className="text-xs text-foreground capitalize">
                      {layer.type} ({Array.isArray(layer.data) ? layer.data.length : 0})
                    </span>
                  </label>
                );
              })}
            </div>
          </div>
        )}

        {/* 개선된 샘플링 경고 */}
        {dataSamplingInfo && (
          <div className="absolute bottom-4 right-4 bg-yellow-50/95 dark:bg-yellow-900/30 backdrop-blur-sm text-yellow-900 dark:text-yellow-100 p-3 rounded-xl shadow-lg border border-yellow-500/30 dark:border-yellow-500/20 text-xs max-w-[240px] animate-in fade-in slide-in-from-bottom-2 duration-200">
            <div className="flex items-start gap-2">
              <span className="text-base">⚠️</span>
              <div className="flex-1">
                <p className="font-semibold mb-1">데이터 샘플링됨</p>
                <p className="text-[11px] opacity-90">
                  {dataSamplingInfo.original.toLocaleString()}개 → {dataSamplingInfo.sampled.toLocaleString()}개 표시
                </p>
                <p className="mt-1 text-[10px] opacity-70">
                  성능 최적화를 위해 샘플링됨
                </p>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
