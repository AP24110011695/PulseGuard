export function EmptyState({ title, hint, command }: { title: string; hint?: string; command?: string }) {
  return (
    <div className="state state-empty">
      <h3>{title}</h3>
      {hint && <p>{hint}</p>}
      {command && <code className="hint-code">{command}</code>}
    </div>
  )
}
