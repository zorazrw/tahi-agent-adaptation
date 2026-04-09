type WorkflowCardProps = {
  steps: string[];
  outputFiles: string[][];
  verifiers: string[][];
  includeVerifiers?: boolean;
};

export function WorkflowCard({ steps, outputFiles, verifiers, includeVerifiers = true }: WorkflowCardProps) {
  if (!steps.length) return null;

  return (
    <div className="mt-3 rounded-2xl border border-primary/20 bg-primary-subtle p-5">
      <div className="text-xs font-semibold text-primary">Workflow Plan</div>
      <div className="mt-3 flex flex-col gap-4">
        {steps.map((step, i) => {
          const files = outputFiles[i] ?? [];
          const criteria = verifiers[i] ?? [];
          return (
            <div key={i} className="flex flex-col gap-1.5">
              <div className="text-sm font-medium text-ink-700">
                {i + 1}. {step}
              </div>
              {files.map((file, fi) => (
                <div key={fi} className="ml-5 flex items-center gap-1.5">
                  <span className="text-xs">📄</span>
                  <span className="rounded bg-ink-900/5 px-2 py-0.5 font-mono text-xs text-ink-700">
                    {file}
                  </span>
                </div>
              ))}
              {includeVerifiers && criteria.map((c, ci) => (
                <div key={ci} className="ml-5 flex items-start gap-1.5 text-xs text-ink-600">
                  <span className="mt-0.5 shrink-0">○</span>
                  <span>{c}</span>
                </div>
              ))}
            </div>
          );
        })}
      </div>
    </div>
  );
}
