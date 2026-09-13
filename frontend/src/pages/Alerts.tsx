import { useState } from 'react'
import { api } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { ErrorState } from '../components/ErrorState'
import { LoadingSkeleton } from '../components/LoadingSkeleton'
import { StatusBadge } from '../components/StatusBadge'
import { useFetch } from '../hooks/useFetch'

export function Alerts() {
  const [filter, setFilter] = useState<'all' | 'unack'>('unack')
  const alerts = useFetch(
    () => api.getAlerts(200),
    [],
  )

  async function acknowledge(id: number) {
    await api.ackAlert(id)
    alerts.retry()
  }

  const rows = alerts.data
    ? filter === 'unack'
      ? alerts.data.alerts.filter((a) => !a.acknowledged)
      : alerts.data.alerts
    : []

  return (
    <section>
      <div className="alerts-head">
        <h2>Alerts</h2>
        <div className="filter-toggle">
          <button
            type="button"
            className={filter === 'unack' ? 'button active' : 'button'}
            onClick={() => setFilter('unack')}
          >
            Unacknowledged
          </button>
          <button
            type="button"
            className={filter === 'all' ? 'button active' : 'button'}
            onClick={() => setFilter('all')}
          >
            All
          </button>
        </div>
      </div>

      {alerts.loading && <LoadingSkeleton height={180} />}
      {alerts.error && <ErrorState message={alerts.error.message} onRetry={alerts.retry} />}
      {alerts.data && rows.length === 0 && (
        <EmptyState
          title="No alerts"
          hint={
            filter === 'unack'
              ? 'All alerts have been acknowledged.'
              : 'Alerts appear when anomalies are flagged, forecasts breach their band, or drift fires.'
          }
        />
      )}
      {rows.length > 0 && (
        <table className="data-table">
          <thead>
            <tr>
              <th>created at</th>
              <th>series</th>
              <th>kind</th>
              <th>severity</th>
              <th>message</th>
              <th>status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {rows.map((a) => (
              <tr key={a.id} className={a.acknowledged ? 'acknowledged' : ''}>
                <td className="muted">{a.created_at.slice(0, 19).replace('T', ' ')}</td>
                <td>{a.series_id}</td>
                <td>{a.kind}</td>
                <td>
                  <StatusBadge label={a.severity} />
                </td>
                <td>{a.message}</td>
                <td>
                  {a.acknowledged ? <StatusBadge label="ack" tone="ok" /> : <StatusBadge label="open" tone="warning" />}
                </td>
                <td>
                  {!a.acknowledged && (
                    <button type="button" className="button" onClick={() => acknowledge(a.id)}>
                      Ack
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}
