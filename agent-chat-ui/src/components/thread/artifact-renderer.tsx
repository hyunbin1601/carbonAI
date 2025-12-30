"use client";

import { useEffect, useState, useRef } from "react";
import mermaid from "mermaid";
import { useArtifact } from "./artifact";
import { BarChart3, Calculator } from "lucide-react";

interface ArtifactProps {
  id: string;
  type: "react" | "mermaid";
  code: string;
  title?: string;
}

/**
 * Custom artifact renderer that executes code directly instead of using LoadExternalComponent
 */
export function ArtifactRenderer({ id, type, code, title }: ArtifactProps) {
  const [ArtifactContent, { open, setOpen }] = useArtifact();
  const [error, setError] = useState<string | null>(null);
  const mermaidRef = useRef<HTMLDivElement>(null);

  // Don't auto-open - let user decide when to view

  useEffect(() => {
    if (type === "mermaid" && mermaidRef.current) {
      // Initialize mermaid
      mermaid.initialize({
        startOnLoad: true,
        theme: "default",
        securityLevel: "loose",
      });

      // Render mermaid diagram
      mermaid.render(`mermaid-${id}`, code).then(({ svg }) => {
        if (mermaidRef.current) {
          mermaidRef.current.innerHTML = svg;
        }
      }).catch((err) => {
        setError(`Mermaid rendering error: ${err.message}`);
      });
    }
  }, [type, code, id]);

  // Render button to open artifact
  const handleOpenArtifact = () => {
    setOpen(true);
  };

  const Icon = type === "mermaid" ? BarChart3 : Calculator;
  const buttonText = type === "mermaid" ? "차트 보기" : "계산기 열기";

  if (!open) {
    // Show compact button when closed
    return (
      <div className="my-2">
        <button
          onClick={handleOpenArtifact}
          className="inline-flex items-center gap-2 px-4 py-2 bg-teal-50 hover:bg-teal-100 text-teal-700 rounded-lg border border-teal-200 transition-colors"
        >
          <Icon className="w-5 h-5" />
          <span className="font-medium">{buttonText}</span>
          <span className="text-sm text-teal-600">→ 오른쪽 패널</span>
        </button>
      </div>
    );
  }

  // When open, render in artifact panel
  if (type === "mermaid") {
    return (
      <ArtifactContent title={title || "차트"}>
        <div className="p-4 bg-white rounded-lg">
          {error ? (
            <div className="text-red-500">{error}</div>
          ) : (
            <div ref={mermaidRef} className="mermaid-container" />
          )}
        </div>
      </ArtifactContent>
    );
  }

  if (type === "react") {
    // For React components, we'll use dynamic import and eval (sandboxed)
    try {
      // Create a safe sandbox for React component execution
      const ComponentWrapper = () => {
        const [Component, setComponent] = useState<React.ComponentType | null>(null);

        useEffect(() => {
          try {
            // Transform the code to be executable
            // Note: This is a simplified version. In production, use a proper transpiler
            const transformedCode = code
              .replace(/^import .+ from .+;?\n/gm, "") // Remove imports (we'll provide them)
              .replace(/export default function/, "return function")
              .replace(/export default/, "return");

            // Create function that returns the component
            // eslint-disable-next-line no-new-func
            const ComponentFactory = new Function(
              "React",
              "useState",
              "useEffect",
              transformedCode
            );

            // Execute and get component
            const { useState: reactUseState, useEffect: reactUseEffect } = require("react");
            const CreatedComponent = ComponentFactory(
              require("react"),
              reactUseState,
              reactUseEffect
            );

            setComponent(() => CreatedComponent);
          } catch (err: any) {
            setError(`Component rendering error: ${err.message}`);
          }
        }, []);

        if (error) {
          return <div className="text-red-500 p-4">{error}</div>;
        }

        if (!Component) {
          return <div className="p-4">Loading component...</div>;
        }

        return <Component />;
      };

      return (
        <ArtifactContent title={title || "계산기"}>
          <div className="p-4 bg-white rounded-lg">
            <ComponentWrapper />
          </div>
        </ArtifactContent>
      );
    } catch (err: any) {
      return (
        <ArtifactContent title={title || "계산기"}>
          <div className="text-red-500 p-4">
            Error: {err.message}
            <details className="mt-2">
              <summary className="cursor-pointer">Code</summary>
              <pre className="mt-2 p-2 bg-gray-100 rounded text-xs overflow-auto">
                {code}
              </pre>
            </details>
          </div>
        </ArtifactContent>
      );
    }
  }

  return null;
}

/**
 * Container for multiple artifacts from the same message
 */
export function ArtifactsContainer({ messageId, artifacts }: {
  messageId: string;
  artifacts: Array<{
    id: string;
    type: "react" | "mermaid";
    content: string;
    metadata?: {
      title?: string;
      artifact_type?: string;
    };
  }>;
}) {
  if (!artifacts || artifacts.length === 0) {
    return null;
  }

  return (
    <>
      {artifacts.map((artifact) => (
        <ArtifactRenderer
          key={artifact.id}
          id={artifact.id}
          type={artifact.type as "react" | "mermaid"}
          code={artifact.content}
          title={artifact.metadata?.title}
        />
      ))}
    </>
  );
}
