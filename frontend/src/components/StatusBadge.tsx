const COLORS: Record<string, string> = {
  ok: '#047857',
  warn: '#b45309',
  drift: '#b91c1c',
  promoted: '#4338ca',
  rejected: '#b91c1c',
  critical: '#b91c1c',
  warning: '#b45309',
  info: '#4338ca',
  loaded: '#047857',
  unloaded: '#6b7280',
}

export function StatusBadge({ label, tone }: { label: string; tone?: string }) {
  const color = COLORS[tone ?? label] ?? '#6b7280'
  return (
    <span className="badge" style={{ color, borderColor: color, background: 'transparent' }}>
      {label}
    </span>
  )
}
