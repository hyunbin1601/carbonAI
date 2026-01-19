"use client";

import { useEffect, useState, useMemo, useRef } from "react";
import Map from "react-map-gl/maplibre";
import DeckGL from "@deck.gl/react";
import { ScatterplotLayer, PathLayer, PolygonLayer, GeoJsonLayer } from "@deck.gl/layers";
import { HexagonLayer } from "@deck.gl/aggregation-layers";
import type { Layer, PickingInfo } from "@deck.gl/core";
import { cn } from "@/lib/utils";
import "maplibre-gl/dist/maplibre-gl.css";

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

  // DeckGL 인스턴스 ref
  const deckRef = useRef<any>(null);

  // 클라이언트 마운트 체크
  useEffect(() => {
    setIsMounted(true);

    // cleanup: 컴포넌트 언마운트 시 WebGL 리소스 정리
    return () => {
      if (deckRef.current) {
        try {
          deckRef.current.finalize();
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

    try {
      let parsedConfig: MapConfig;

      if (typeof config === "string") {
        parsedConfig = JSON.parse(config);
      } else {
        parsedConfig = config;
      }

      setMapConfig(parsedConfig);
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

    const createdLayers = mapConfig.layers.map((layerConfig, index) => {
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
            getFillColor: (d: any) => d.color || [255, 140, 0, 180],
            radiusMinPixels: layerConfig.radiusMinPixels || 5,
            radiusMaxPixels: layerConfig.radiusMaxPixels || 30,
          });

        case "path":
          return new PathLayer({
            ...commonProps,
            getPath: (d: any) => d.path,
            getColor: (d: any) => d.color || [255, 0, 0, 200],
            getWidth: (d: any) => d.width || 5,
            widthMinPixels: layerConfig.widthMinPixels || 2,
          });

        case "polygon":
          return new PolygonLayer({
            ...commonProps,
            getPolygon: (d: any) => d.polygon,
            getFillColor: (d: any) => d.fillColor || [0, 200, 0, 100],
            getLineColor: (d: any) => d.lineColor || [0, 0, 0, 255],
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
            getFillColor: (d: any) => d.properties?.fillColor || [160, 160, 180, 200],
            getLineColor: (d: any) => d.properties?.lineColor || [0, 0, 0, 255],
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
  }, [mapConfig]);

  // 툴팁 정보
  const handleHover = (info: PickingInfo) => {
    setHoveredObject(info.object);
  };

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

  const viewState = {
    ...DEFAULT_VIEW_STATE,
    ...mapConfig.initialViewState,
  };

  const mapStyle = mapConfig.style || (isDarkMode ? DARK_MAP_STYLE : LIGHT_MAP_STYLE);

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
      className={cn(
        "map-container rounded-xl overflow-hidden border border-border/50 dark:border-zinc-700/50 shadow-sm",
        className
      )}
    >
      <div className="relative w-full h-[500px]">
        <DeckGL
          ref={deckRef}
          initialViewState={viewState}
          controller={true}
          layers={mapLoaded ? layers : []}
          onHover={handleHover}
          getTooltip={() => null}
          useDevicePixels={1}
        >
          <Map
            mapStyle={mapStyle}
            onLoad={() => {
              setTimeout(() => setMapLoaded(true), 100);
            }}
          />
        </DeckGL>

        {/* 툴팁 */}
        {mapConfig.tooltip !== false && hoveredObject && (
          <div
            className="absolute bottom-4 left-4 bg-white dark:bg-zinc-800 p-3 rounded-lg shadow-lg border border-border/50 max-w-xs"
            style={{ pointerEvents: "none" }}
          >
            <pre className="text-xs text-foreground overflow-x-auto">
              {JSON.stringify(hoveredObject, null, 2)}
            </pre>
          </div>
        )}

        {/* 샘플링 경고 */}
        {dataSamplingInfo && (
          <div className="absolute top-4 right-4 bg-yellow-100 dark:bg-yellow-900/50 text-yellow-900 dark:text-yellow-100 p-2 rounded-lg shadow-lg border border-yellow-500/30 text-xs">
            <p className="font-semibold">⚠️ 데이터 샘플링됨</p>
            <p className="mt-1">
              원본: {dataSamplingInfo.original.toLocaleString()}개 → 표시: {dataSamplingInfo.sampled.toLocaleString()}개
            </p>
            <p className="mt-1 text-[10px] opacity-80">
              성능을 위해 데이터가 샘플링되었습니다
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
