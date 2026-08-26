const ACTIVE_CONTENT = /<(script|iframe|object|embed|foreignObject)\b/i;
const EVENT_HANDLER = /\son[a-z][\w:-]*\s*=/i;
const JAVASCRIPT_URL = /(?:href|src)\s*=\s*["']\s*javascript:/i;
const XML_DECLARATION = /^<\?xml\s+version\s*=\s*["'][^"']+["'](?:\s+encoding\s*=\s*["'][^"']+["'])?\s*\?>\s*/i;
const DOCUMENT_DIRECTIVE = /<!DOCTYPE|<\?xml\b/i;
const RESOURCE_ATTRIBUTE = /(?:href|src)\s*=\s*(["'])(.*?)\1/gi;
const CSS_URL = /url\(\s*(["']?)(.*?)\1\s*\)/gi;
const CSS_IMPORT = /@import\s+(?:url\s*\()?\s*["']?[^;\s)'"]+/i;
const EMBEDDED_RASTER = /^data:image\/(?:png|gif|jpe?g|webp);base64,/i;

function isSafeResource(value: string): boolean {
  const trimmed = value.trim();
  return trimmed.startsWith("#") || EMBEDDED_RASTER.test(trimmed);
}

function assertSelfContained(svg: string): void {
  if (CSS_IMPORT.test(svg)) {
    throw new Error("External resources are not allowed in SVG documents.");
  }
  for (const match of svg.matchAll(RESOURCE_ATTRIBUTE)) {
    if (!isSafeResource(match[2] ?? "")) {
      throw new Error("External resources are not allowed in SVG documents.");
    }
  }
  for (const match of svg.matchAll(CSS_URL)) {
    if (!isSafeResource(match[2] ?? "")) {
      throw new Error("External resources are not allowed in SVG documents.");
    }
  }
}

export function validateSvg(svg: string): string {
  const trimmed = svg.trim().replace(XML_DECLARATION, "");
  if (!/^<svg(?:\s|>)/i.test(trimmed) || !/<\/svg>$/i.test(trimmed)) {
    throw new Error("The document must be a complete <svg> element.");
  }
  if (ACTIVE_CONTENT.test(trimmed)) {
    throw new Error("Unsafe embedded content is not allowed in SVG documents.");
  }
  if (EVENT_HANDLER.test(trimmed) || JAVASCRIPT_URL.test(trimmed)) {
    throw new Error("Event handlers and JavaScript URLs are not allowed in SVG documents.");
  }
  if (DOCUMENT_DIRECTIVE.test(trimmed)) {
    throw new Error("Document directives are not allowed in SVG documents.");
  }
  assertSelfContained(trimmed);
  return trimmed;
}
