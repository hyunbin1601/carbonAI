import { parsePartialJson } from "@langchain/core/output_parsers";
import { useStreamContext } from "@/hooks/useStreamContext";
import { AIMessage, Checkpoint, Message } from "@langchain/langgraph-sdk";
import { getContentString } from "../utils";
import { BranchSwitcher, CommandBar } from "./shared";
import { MarkdownText } from "../markdown-text";
import { LoadExternalComponent } from "@langchain/langgraph-sdk/react-ui";
import { cn } from "@/lib/utils";
import { ToolCalls, ToolResult } from "./tool-calls";
import { MessageContentComplex } from "@langchain/core/messages";
import { Fragment } from "react/jsx-runtime";
import { isAgentInboxInterruptSchema } from "@/lib/agent-inbox-interrupt";
import { ThreadView } from "../agent-inbox";
import { useQueryState, parseAsBoolean } from "nuqs";
import { GenericInterruptView } from "./generic-interrupt";
import { useArtifact } from "../artifact";
import { useTypingEffect } from "@/hooks/useTypingEffect";
import {
  filterIncompleteVisualizationBlocks,
  getVisualizationLoadingMessage,
} from "@/lib/code-block-filter";

function CustomComponent({
  message,
  thread,
}: {
  message: Message;
  thread: ReturnType<typeof useStreamContext>;
}) {
  const artifact = useArtifact();
  const { values } = useStreamContext();
  const customComponents = values.ui?.filter(
    (ui) => ui.metadata?.message_id === message.id,
  );

  if (!customComponents?.length) return null;
  return (
    <Fragment key={message.id}>
      {customComponents.map((customComponent) => (
        <LoadExternalComponent
          key={customComponent.id}
          stream={thread}
          message={customComponent}
          meta={{ ui: customComponent, artifact }}
        />
      ))}
    </Fragment>
  );
}

interface ToolUseContent {
  type: "tool_use";
  id: string;
  name?: string;
  input?: string | object;
}

function isToolUseContent(content: MessageContentComplex): content is ToolUseContent {
  return content.type === "tool_use" && "id" in content;
}

function parseAnthropicStreamedToolCalls(
  content: MessageContentComplex[],
): AIMessage["tool_calls"] {
  const toolCallContents = content.filter(isToolUseContent);

  return toolCallContents.map((tc) => {
    let args: Record<string, unknown> = {};
    if (tc.input) {
      try {
        const parsedInput = typeof tc.input === "string"
          ? parsePartialJson(tc.input)
          : tc.input;
        args = parsedInput ?? {};
      } catch {
        // Pass
      }
    }
    return {
      name: tc.name ?? "",
      id: tc.id,
      args,
      type: "tool_call" as const,
    };
  });
}

interface InterruptProps {
  interruptValue?: unknown;
  isLastMessage: boolean;
  hasNoAIOrToolMessages: boolean;
}

function Interrupt({
  interruptValue,
  isLastMessage,
  hasNoAIOrToolMessages,
}: InterruptProps) {
  return (
    <>
      {isAgentInboxInterruptSchema(interruptValue) &&
        (isLastMessage || hasNoAIOrToolMessages) && (
          <ThreadView interrupt={interruptValue} />
        )}
      {interruptValue &&
      !isAgentInboxInterruptSchema(interruptValue) &&
      (isLastMessage || hasNoAIOrToolMessages) ? (
        <GenericInterruptView interrupt={interruptValue as Record<string, unknown> | Record<string, unknown>[]} />
      ) : null}
    </>
  );
}

export function AssistantMessage({
  message,
  isLoading,
  handleRegenerate,
}: {
  message: Message | undefined;
  isLoading: boolean;
  handleRegenerate: (parentCheckpoint: Checkpoint | null | undefined) => void;
}) {
  const content = message?.content ?? [];
  const contentString = getContentString(content);
  const [hideToolCalls] = useQueryState(
    "hideToolCalls",
    parseAsBoolean.withDefault(true),
  );

  const thread = useStreamContext();
  const isLastMessage =
    thread.messages[thread.messages.length - 1].id === message?.id;

  // 타이핑 효과: 마지막 AI 메시지이고 로딩 중이거나 방금 완료된 경우에만 적용
  const shouldAnimate = isLastMessage && message?.type === "ai";
  const { displayedText, isTyping } = useTypingEffect(contentString, {
    speed: 3, // 글자당 3ms (매우 빠른 타이핑 - 이전 10ms에서 3배 향상)
    enabled: shouldAnimate,
  });
  const hasNoAIOrToolMessages = !thread.messages.find(
    (m) => m.type === "ai" || m.type === "tool",
  );
  const meta = message ? thread.getMessagesMetadata(message) : undefined;
  const threadInterrupt = thread.interrupt;

  const parentCheckpoint = meta?.firstSeenState?.parent_checkpoint;
  const anthropicStreamedToolCalls = Array.isArray(content)
    ? parseAnthropicStreamedToolCalls(content)
    : undefined;

  const hasToolCalls =
    message &&
    "tool_calls" in message &&
    message.tool_calls &&
    message.tool_calls.length > 0;
  const toolCallsHaveContents =
    hasToolCalls &&
    message.tool_calls?.some(
      (tc) => tc.args && Object.keys(tc.args).length > 0,
    );
  const hasAnthropicToolCalls = !!anthropicStreamedToolCalls?.length;
  const isToolResult = message?.type === "tool";

  if (isToolResult && hideToolCalls) {
    return null;
  }

  return (
    <div className="group mr-auto flex items-start gap-3">
      <div className="flex flex-col gap-3">
        {isToolResult ? (
          <>
            <ToolResult message={message} isLoading={isLoading} />
            <Interrupt
              interruptValue={threadInterrupt?.value}
              isLastMessage={isLastMessage}
              hasNoAIOrToolMessages={hasNoAIOrToolMessages}
            />
          </>
        ) : (
          <>
            {displayedText && (() => {
              // 불완전한 시각화 코드 블록 필터링
              const { filteredText, pendingLanguage } = filterIncompleteVisualizationBlocks(displayedText);

              return (
                <>
                  {filteredText && (
                    <div className="py-1 leading-relaxed">
                      <MarkdownText>{filteredText}</MarkdownText>
                      {/* 타이핑 커서 표시 - 시각화 생성 중이 아닐 때만 */}
                      {isTyping && !pendingLanguage && (
                        <span className="inline-block w-0.5 h-4 bg-foreground/70 animate-pulse ml-0.5 align-middle" />
                      )}
                    </div>
                  )}
                  {/* 시각화 생성 중 로딩 표시 */}
                  {pendingLanguage && (
                    <div className="rounded-xl bg-muted/50 dark:bg-zinc-900 p-6 border border-border/30 dark:border-zinc-700 animate-pulse">
                      <div className="flex items-center justify-center gap-3 text-muted-foreground">
                        <div className="flex gap-1">
                          <div className="h-2 w-2 rounded-full bg-foreground/40 animate-[pulse_1.5s_ease-in-out_infinite]" />
                          <div className="h-2 w-2 rounded-full bg-foreground/40 animate-[pulse_1.5s_ease-in-out_0.3s_infinite]" />
                          <div className="h-2 w-2 rounded-full bg-foreground/40 animate-[pulse_1.5s_ease-in-out_0.6s_infinite]" />
                        </div>
                        <span className="text-sm">{getVisualizationLoadingMessage(pendingLanguage)}</span>
                      </div>
                    </div>
                  )}
                </>
              );
            })()}

            {!hideToolCalls && (
              <>
                {(hasToolCalls && toolCallsHaveContents && (
                  <ToolCalls toolCalls={message.tool_calls} isLoading={isLoading} />
                )) ||
                  (hasAnthropicToolCalls && (
                    <ToolCalls toolCalls={anthropicStreamedToolCalls} isLoading={isLoading} />
                  )) ||
                  (hasToolCalls && (
                    <ToolCalls toolCalls={message.tool_calls} isLoading={isLoading} />
                  ))}
              </>
            )}

            {message && (
              <CustomComponent
                message={message}
                thread={thread}
              />
            )}
            <Interrupt
              interruptValue={threadInterrupt?.value}
              isLastMessage={isLastMessage}
              hasNoAIOrToolMessages={hasNoAIOrToolMessages}
            />
            <div
              className={cn(
                "mr-auto flex items-center gap-2 transition-opacity",
                "opacity-0 group-focus-within:opacity-100 group-hover:opacity-100",
              )}
            >
              <BranchSwitcher
                branch={meta?.branch}
                branchOptions={meta?.branchOptions}
                onSelect={(branch) => thread.setBranch(branch)}
                isLoading={isLoading}
              />
              <CommandBar
                content={contentString}
                isLoading={isLoading}
                isAiMessage={true}
                handleRegenerate={() => handleRegenerate(parentCheckpoint)}
              />
            </div>
          </>
        )}
      </div>
    </div>
  );
}

export function AssistantMessageLoading() {
  return (
    <div className="mr-auto flex items-start gap-3">
      <div className="bg-muted flex h-9 items-center gap-1.5 rounded-2xl px-5 py-2.5 shadow-sm border border-border/20">
        <div className="bg-foreground/40 h-1.5 w-1.5 animate-[pulse_1.5s_ease-in-out_infinite] rounded-full"></div>
        <div className="bg-foreground/40 h-1.5 w-1.5 animate-[pulse_1.5s_ease-in-out_0.5s_infinite] rounded-full"></div>
        <div className="bg-foreground/40 h-1.5 w-1.5 animate-[pulse_1.5s_ease-in-out_1s_infinite] rounded-full"></div>
      </div>
    </div>
  );
}
