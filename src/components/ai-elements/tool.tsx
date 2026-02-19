"use client";

import type { ComponentProps, ReactNode } from "react";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/ui/components/ui/collapsible";
import { cn } from "@/lib/utils";
import {
  ChevronRightIcon,
  LoaderIcon,
} from "lucide-react";

export type ToolState = "pending" | "running" | "completed" | "error";

export type ToolProps = ComponentProps<typeof Collapsible>;

export const Tool = ({ className, ...props }: ToolProps) => (
  <Collapsible
    className={cn("group not-prose w-full rounded-lg border border-ink-900/8", className)}
    {...props}
  />
);

export type ToolHeaderProps = ComponentProps<typeof CollapsibleTrigger> & {
  icon?: ReactNode;
  title: string;
  description?: string;
  suffix?: ReactNode;
  state: ToolState;
};

export const ToolHeader = ({
  className,
  icon,
  title,
  description,
  suffix,
  state,
  ...props
}: ToolHeaderProps) => (
  <CollapsibleTrigger
    className={cn(
      "flex w-full items-center gap-2 px-3 py-2 text-sm",
      className
    )}
    {...props}
  >
    {icon && <span className="shrink-0 text-muted-foreground">{icon}</span>}
    <span className="font-semibold shrink-0">{title}</span>
    {description && (
      <span className="truncate text-muted-foreground font-mono text-xs">{description}</span>
    )}
    {state === "running" && (
      <LoaderIcon className="size-3.5 shrink-0 text-muted-foreground animate-spin" />
    )}
    <span className="flex-1" />
    {suffix}
    <ChevronRightIcon className="size-3.5 shrink-0 text-muted-foreground transition-transform duration-150 group-data-[state=open]:rotate-90" />
  </CollapsibleTrigger>
);

export type ToolContentProps = ComponentProps<typeof CollapsibleContent>;

export const ToolContent = ({ className, ...props }: ToolContentProps) => (
  <CollapsibleContent
    className={cn("border-t border-ink-900/8 text-sm", className)}
    {...props}
  />
);
