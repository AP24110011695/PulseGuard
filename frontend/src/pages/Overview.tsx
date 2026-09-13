import { api } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { ErrorState } from '../components/ErrorState'
import { LoadingSkeleton } from '../components/LoadingSkeleton'
import { StatusBadge } from '../components/StatusBadge'
import { useFetch } from '../hooks/useFetch'

export function Overview({ onSelect }: { onSelect: (id: number) => void }) {
  const series = useFetch(api.listSeries, [])
  const champions = useFetch(
    () => Promise.all([api.getChampion('forecast'), api.getChampion('anomaly')]),
    [],
  )
  const alerts = useFetch(() => api.getAlerts(200), [])
  const drift = useFetch(api.getDriftStatus, [])

  if (series.loading) return <LoadingSkeleton height={200} />
  if (series.error) return <ErrorState message={series.error.message} onRetry={series.retry} />
  if (!series.data || series.data.length === 0) {
    return (
      <EmptyState
        title="No metric series yet"
        hint="Generate and ingest synthetic metrics with the simulator:"
        command="python -m ml.simulator --config ml/configs/simulator_default.yaml --ingest"
      />
    )
  }

  const openAlerts = alerts.data?.unacknowledged ?? 0
  const driftCount = drift.data?.series.filter((s) => s.overall !== 'ok').length ?? 0

  return (
    <section>
      <h2>System overview</h2>
      <div className="summary-grid">
        <div className="card summary-card">
          <span className="muted small">forecast champion</span>
          {champions.loading ? (
            <span className="muted">…</span>
          ) : (
            <span className="series-name">
              v{champions.data?.[0]?.model_version}{' '}
              <StatusBadge
                label={champions.data?.[0]?.loaded ? 'loaded' : 'unloaded'}
                tone={champions.data?.[0]?.loaded ? 'loaded' : 'unloaded'}
              />
            </span>
          )}
        </div>
        <div className="card summary-card">
          <span className="muted small">anomaly champion</span>
          {champions.loading ? (
            <span className="muted">…</span>
          ) : (
            <span className="series-name">
              v{champions.data?.[1]?.model_version}{' '}
              <StatusBadge
                label={champions.data?.[1]?.loaded ? 'loaded' : 'unloaded'}
                tone={champions.data?.[1]?.loaded ? 'loaded' : 'unloaded'}
              />
            </span>
          )}
        </div>
        <div className="card summary-card">
          <span className="muted small">drift status</span>
          <span>
            {drift.loading ? (
              '…'
            ) : (
              <StatusBadge label={driftCount > 0 ? `${driftCount} series off-nominal` : 'all ok'} tone={driftCount > 0 ? 'warn' : 'ok'} />
            )}
          </span>
        </div>
        <div className="card summary-card">
          <span className="muted small">open alerts</span>
          <span className={openAlerts > 0 ? 'alert-count' : ''}>{openAlerts}</span>
        </div>
      </div>

      <h2>Metric series</h2>
      <div className="card-grid">
        {series.data.map((s) => (
          <button key={s.id} type="button" className="card series-card" onClick={() => onSelect(s.id)}>
            <div className="series-card-head">
              <span className="series-name">{s.name}</span>
              <span className={`badge badge-${s.source}`}>{s.source}</span>
            </div>
            {s.description && <p className="muted">{s.description}</p>}
            <div className="series-card-foot">
              {s.unit && <span className="muted">unit: {s.unit}</span>}
              <span className="muted">id: {s.id}</span>
            </div>
          </button>
        ))}
      </div>
    </section>
  )
}
