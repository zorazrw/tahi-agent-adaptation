export function VideoRenderer({ data }: { data: { kind: "video"; dataUrl: string } }) {
  return (
    <div className="flex items-center justify-center h-full min-h-0">
      <video
        src={data.dataUrl}
        controls
        className="max-w-full max-h-full rounded"
      />
    </div>
  );
}
