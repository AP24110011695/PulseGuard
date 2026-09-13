import { api } from '../api/client'
import type { ChampionInfo, PromotionDecision } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { ErrorState } from '../components/ErrorState'
import { LoadingSkeleton } from '../components/LoadingSkeleton'
import { MetricsTable } from '../components/MetricsTable'
import { StatusBadge } from '../components/StatusBadge'
import { useFetch } from '../hooks/useFetch'

function fmt(value: unknown): string {
  if (value === null || value === undefined) return '—'
  if (typeof value === 'number') return value.toFixed(4)
  if (Array.isArray(value)) return `[${value.join(', ')}]`
  return String(value)
}

function ChampionCard({ task, champion }: { task: string; champion: ChampionInfo | null }) {
  if (!champion) {
    return (
      <div className="card">
        <h3>
          {task} champion <StatusBadge label="none" tone="unloaded" />
        </h3>
        <p className="muted">No champion registered yet.</p>
      </div>
    )
  }
  const metrics = (champion.metrics ?? {}) as Record<string, unknown>
  return (
    <div className="card">
      <h3>
        {task} champion <StatusBadge label={champion.loaded ? 'loaded' : 'unloaded'} />
      </h3>
      <MetricsTable
        rows={[
          { label: 'model', value: `${champion.model_name} v${champion.model_version}` },
          { label: 'promoted at', value: champion.promoted_at?.slice(0, 19).replace('T', ' ') ?? '—' },
          { label: 'horizons', value: fmt(metrics.horizons) },
          { label: 'quantiles', value: fmt(metrics.quantiles) },
          { label: 'threshold', value: fmt(metrics.threshold) },
        ]}
      />
    </div>
  )
}

function PromotionsTable({ decisions }: { decisions: PromotionDecision[] }) {
  if (decisions.length === 0) {
    return <EmptyState title="No promotion decisions yet" hint="Run the drift demo or trigger retraining." />
  }
  return (
    <table className="data-table">
      <thead>
        <tr>
          <th>decided at</th>
          <th>task</th>
          <th>champion</th>
          <th>challenger</th>
          <th>decision</th>
          <th>reason</th>
        </tr>
      </thead>
      <tbody>
        {decisions.map((d) => (
          <tr key={d.id}>
            <td className="muted">{d.decided_at.slice(0, 19).replace('T', ' ')}</td>
            <td>{d.task}</td>
            <td>
              {d.champion_name.split('-').pop()} v{d.champion_version}
            </td>
            <td>
              {d.challenger_name.split('-').pop()}
              {d.challenger_version !== null ? ` v${d.challenger_version}` : ''}
            </td>
            <td>
              <StatusBadge label={d.decision} />
            </td>
            <td className="muted">{d.reason}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

export function Models() {
  const champions = useFetch(
    () => Promise.all([api.getChampion('forecast'), api.getChampion('anomaly')]),
    [],
  )
  const models = useFetch(api.listRegisteredModels, [])
  const promotions = useFetch(() => api.getPromotions(25), [])

  return (
    <section>
      <h2>Model registry</h2>
      {champions.loading && <LoadingSkeleton height={140} />}
      {champions.error && <ErrorState message={champions.error.message} onRetry={champions.retry} />}
      {champions.data && (
        <div className="card-grid">
          <ChampionCard task="forecast" champion={champions.data[0]} />
          <ChampionCard task="anomaly" champion={champions.data[1]} />
        </div>
      )}

      <h2>Registered versions</h2>
      {models.loading && <LoadingSkeleton height={160} />}
      {models.error && <ErrorState message={models.error.message} onRetry={models.retry} />}
      {models.data && models.data.length > 0 && (
        <table className="data-table">
          <thead>
            <tr>
              <th>model</th>
              <th>version</th>
              <th>status</th>
              <th>aliases</th>
              <th>run</th>
            </tr>
          </thead>
          <tbody>
            {models.data.map((m) =>
              m.versions.map((v) => (
                <tr key={`${m.name}-${v.version}`}>
                  <td>{m.name}</td>
                  <td>v{v.version}</td>
                  <td>
                    <StatusBadge label={v.status} tone="info" />
                  </td>
                  <td>{v.aliases.length > 0 ? <StatusBadge label={v.aliases.join(', ')} tone="promoted" /> : '—'}</td>
                  <td className="muted">{v.run_id ? `${v.run_id.slice(0, 8)}…` : '—'}</td>
                </tr>
              )),
            )}
          </tbody>
        </table>
      )}

      <h2>Promotion decisions</h2>
      {promotions.loading && <LoadingSkeleton height={160} />}
      {promotions.error && <ErrorState message={promotions.error.message} onRetry={promotions.retry} />}
      {promotions.data && <PromotionsTable decisions={promotions.data.decisions} />}
    </section>
  )
}
