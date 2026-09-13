export function LoadingSkeleton({ height = 220 }: { height?: number }) {
  return <div className="skeleton" style={{ height }} aria-label="Loading" />
}
