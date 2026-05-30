/**
 * Strip Kimi K2 wire-format tool-call tokens that sometimes remain in assistant
 * ``text`` when the renderer leaves serialization in the content channel.
 */

const KIMI_TOOL_SECTION_RE =
  /<\|redacted_tool_calls_section_begin\|>[\s\S]*?<\|redacted_tool_calls_section_end\|>/g;
const KIMI_TOOL_CALL_RE =
  /<\|redacted_tool_call_begin_kimi\|>[\s\S]*?<\|redacted_tool_call_end_kimi\|>/g;
/** Unclosed or partial tail (truncated completion). */
const KIMI_TOOL_TAIL_RE = /<\|redacted_tool_call[\s\S]*$/;
/** Orphan closing / section markers. */
const KIMI_TOOL_ORPHAN_RE = /<\|redacted_tool_calls?_section_(?:begin|end)\|>/g;
const KIMI_TOOL_FRAGMENT_RE = /<\|redacted_tool_call[^|]*\|>/g;

/** Trailing JSON close from tool args leaked into text before wire tokens. */
const TRAILING_TOOL_JSON_TAIL_RE = /"\s*\}\s*$/;

/** Write-tool file body often leaked into the text channel after a short title. */
const HTML_PAYLOAD_START_RE = /\n\s*<(?:!DOCTYPE|html|head|body|div|canvas|script)\b/i;
const CHART_JS_PAYLOAD_START_RE = /\n\s*(?:const\s+\w+\s*=|new\s+Chart\s*\(|plugins:\s*\[|scales:\s*\{)/i;

function trimLeakedToolFilePayload(text: string): string {
  let cut = -1;
  const htmlMatch = HTML_PAYLOAD_START_RE.exec(text);
  if (htmlMatch && htmlMatch.index > 0) {
    cut = htmlMatch.index;
  }
  const chartMatch = CHART_JS_PAYLOAD_START_RE.exec(text);
  if (chartMatch && chartMatch.index > 0) {
    cut = cut < 0 ? chartMatch.index : Math.min(cut, chartMatch.index);
  }
  if (cut > 0) {
    const prefix = text.slice(0, cut).trim();
    if (prefix.length > 0) {
      return prefix;
    }
  }
  return text;
}

export function stripKimiToolWireFormatFromAssistantText(text: string): string {
  let t = text;
  t = t.replace(KIMI_TOOL_SECTION_RE, "");
  t = t.replace(KIMI_TOOL_CALL_RE, "");
  t = t.replace(KIMI_TOOL_TAIL_RE, "");
  t = t.replace(KIMI_TOOL_ORPHAN_RE, "");
  t = t.replace(KIMI_TOOL_FRAGMENT_RE, "");
  t = t.replace(TRAILING_TOOL_JSON_TAIL_RE, "");
  return t.trim();
}

export function assistantTextForDisplay(text: string): string {
  const stripped = stripKimiToolWireFormatFromAssistantText(text);
  return trimLeakedToolFilePayload(stripped);
}

export function shouldShowAssistantTextBlock(text: string): boolean {
  return assistantTextForDisplay(text).length > 0;
}
