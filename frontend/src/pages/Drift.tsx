import { useState } from 'react'
import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { api } from '../api/client'
import type { DriftStatusEntry } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { ErrorState } from '../components/ErrorState'
import { LoadingSkeleton } from '../components/LoadingSkeleton'
import { StatusBadge } from '../components/StatusBadge'
import { useFetch } from '../hooks/useFetch'

function PsiBars({ entry }: { entry: DriftStatusEntry }) {
  const psi = entry.psi
  if (!psi) return <p className="muted">No PSI check recorded for this series yet.</p>
  const details = (psi.details ?? {}) as { feature_psis?: Record<string, number> }
  const featurePsis = details.feature_psis ?? {}
  const data = Object.entries(featurePsis)
    .map(([feature, value]) => ({ feature, psi: Number(value.toFixed(4)) }))
    .sort((a, b) => b.psi - a.psi)
    .slice(0, 12)

  return (
    <div className="chart-frame">
      <p className="muted small">
        Latest check {psi.detected_at.slice(0, 19).replace('T', ' ')} UTC — worst feature{' '}
        <strong>{psi.feature}</strong> (PSI {psi.psi_value?.toFixed(3)})
      </p>
      <ResponsiveContainer width="100%" height={260}>
        <BarChart data={data} margin={{ top: 8, right: 16, bottom: 40, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="feature"
            tick={{ fontSize: 10 }}
            stroke="#6b7280"
            angle={-40}
            textAnchor="end"
            interval={0}
          />
          <YAxis tick={{ fontSize: 11 }} stroke="#6b7280" width={48} />
          <Tooltip formatter={(value: number | string) => [value, 'PSI']} />
          <Bar dataKey="psi" fill={psi.status === 'drift' ? '#dc2626' : '#4f46e5'} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

export function Drift() {
  const status = useFetch(api.getDriftStatus, [])
  const events = useFetch(() => api.getDriftEvents(20), [])
  const [selected, setSelected] = useState<number | null>(null)

  return (
    <section>
      <h2>Drift status</h2>
      {status.loading && <LoadingSkeleton height={180} />}
      {status.error && <ErrorState message={status.error.message} onRetry={status.retry} />}
      {status.data && status.data.series.length === 0 && (
        <EmptyState title="No series to monitor" hint="Ingest data and run a drift check first." />
      )}
      {status.data && status.data.series.length > 0 && (
        <div className="card-grid">
          {status.data.series.map((s) => (
            <button
              key={s.series_id}
              type="button"
              className={`card series-card ${selected === s.series_id ? 'selected' : ''}`}
              onClick={() => setSelected(selected === s.series_id ? null : s.series_id)}
            >
              <div className="series-card-head">
                <span className="series-name">{s.series_name}</span>
                <StatusBadge label={s.overall} />
              </div>
              <div className="series-card-foot">
                <span>
                  PSI <StatusBadge label={(s.psi ?? { status: 'none' }).status} />
                </span>
                <span>
                  residual <StatusBadge label={(s.residual ?? { status: 'none' }).status} />
                </span>
              </div>
              {s.psi?.feature && (
                <p className="muted small">
                  worst feature: {s.psi.feature} (PSI {s.psi.psi_value?.toFixed(3)})
                </p>
              )}
            </button>
          ))}
        </div>
      )}

      {status.data && selected !== null && (
        <>
          <h2>Per-feature PSI — {status.data.series.find((s) => s.series_id === selected)?.series_name}</h2>
          <PsiBars entry={status.data.series.find((s) => s.series_id === selected)!} />
        </>
      )}

      <h2>Recent drift events</h2>
      {events.loading && <LoadingSkeleton height={160} />}
      {events.error && <ErrorState message={events.error.message} onRetry={events.retry} />}
      {events.data && events.data.count === 0 && (
        <EmptyState title="No drift events yet" hint="Run a drift check from the API or the demo." />
      )}
      {events.data && events.data.count > 0 && (
        <table className="data-table">
          <thead>
            <tr>
              <th>detected at</th>
              <th>series</th>
              <th>kind</th>
              <th>status</th>
              <th>feature</th>
              <th>value</th>
            </tr>
          </thead>
          <tbody>
            {events.data.events.map((e) => (
              <tr key={e.id}>
                <td className="muted">{e.detected_at.slice(0, 19).replace('T', ' ')}</td>
                <td>{e.series_id}</td>
                <td>{e.kind}</td>
                <td>
                  <StatusBadge label={e.status} />
                </td>
                <td>{e.feature ?? '—'}</td>
                <td>{e.psi_value !== null ? e.psi_value.toFixed(3) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}
