export interface MetricsRow {
  label: string
  value: string
}

export function MetricsTable({ rows }: { rows: MetricsRow[] }) {
  return (
    <table className="metrics-table">
      <tbody>
        {rows.map((r) => (
          <tr key={r.label}>
            <td className="muted">{r.label}</td>
            <td>{r.value}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}
