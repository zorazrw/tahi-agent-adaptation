import { describe, expect, test } from "bun:test";
import { validateSvg } from "../src/lib/svg-tools.js";

function validationError(svg: string): string {
  try {
    validateSvg(svg);
    return "";
  } catch (error) {
    return error instanceof Error ? error.message : String(error);
  }
}

describe("validateSvg", () => {
  test("accepts a complete self-contained SVG", () => {
    const svg = `
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
        <defs><linearGradient id="g"><stop offset="0" stop-color="#fff" /></linearGradient></defs>
        <rect width="100" height="100" fill="url(#g)" />
      </svg>
    `;

    expect(validateSvg(svg)).toBe(svg.trim());
  });

  test("accepts and removes a standard XML declaration", () => {
    expect(validateSvg('<?xml version="1.0" encoding="UTF-8"?><svg></svg>')).toBe("<svg></svg>");
  });

  test("rejects incomplete documents", () => {
    expect(validationError("<rect width=\"10\" />")).toContain("complete <svg>");
  });

  const activeContent = [
    "<svg><script>alert(1)</script></svg>",
    "<svg><foreignObject><div>unsafe</div></foreignObject></svg>",
    "<svg><rect onclick=\"alert(1)\" /></svg>",
    "<svg><a href=\"javascript:alert(1)\"><text>unsafe</text></a></svg>",
    "<!DOCTYPE svg><svg></svg>",
    "<?xml-stylesheet href=\"https://example.com/x.css\"?><svg></svg>",
  ];
  for (const svg of activeContent) {
    test(`rejects active SVG content: ${svg}`, () => {
      expect(validationError(svg)).not.toBe("");
    });
  }

  const externalResources = [
    "<svg><image href=\"https://example.com/pixel.png\" /></svg>",
    "<svg><use href=\"other.svg#shape\" /></svg>",
    "<svg><style>rect { fill: url(https://example.com/pattern.svg) }</style></svg>",
    "<svg><style>@import 'https://example.com/style.css';</style></svg>",
  ];
  for (const svg of externalResources) {
    test(`rejects external resources: ${svg}`, () => {
      expect(validationError(svg)).toContain("External resources");
    });
  }

  test("allows fragment references and embedded raster images", () => {
    expect(validateSvg('<svg><use href="#shape" /><image href="data:image/png;base64,AA==" /></svg>'))
      .toContain("data:image/png");
  });
});
