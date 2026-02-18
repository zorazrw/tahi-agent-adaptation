export function AudioRenderer({ data }: { data: { kind: "audio"; dataUrl: string } }) {
  return (
    <div className="flex items-center justify-center py-8">
      <audio src={data.dataUrl} controls className="w-full max-w-lg" />
    </div>
  );
}
