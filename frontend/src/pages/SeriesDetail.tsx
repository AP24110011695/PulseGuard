import { api } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { ErrorState } from '../components/ErrorState'
import { LoadingSkeleton } from '../components/LoadingSkeleton'
import { MetricChart } from '../components/MetricChart'
import { StatusBadge } from '../components/StatusBadge'
import { useFetch } from '../hooks/useFetch'

function fmtRange(first: string | null, last: string | null): string {
  if (!first || !last) return 'no points'
  return `${first.slice(0, 16).replace('T', ' ')} → ${last.slice(0, 16).replace('T', ' ')} UTC`
}

export function SeriesDetail({ id, onBack }: { id: number; onBack: () => void }) {
  const detail = useFetch(() => api.getSeries(id), [id])
  const points = useFetch(() => api.getPoints(id), [id])
  const forecasts = useFetch(() => api.getStoredForecasts(id), [id])
  const anomalies = useFetch(() => api.getStoredAnomalies(id), [id])

  return (
    <section>
      <button type="button" className="button button-link" onClick={onBack}>
        ← All series
      </button>

      {detail.loading && <LoadingSkeleton height={90} />}
      {detail.error && <ErrorState message={detail.error.message} onRetry={detail.retry} />}
      {detail.data && (
        <div className="detail-head">
          <h2>
            {detail.data.name}
            {detail.data.unit && <span className="unit"> ({detail.data.unit})</span>}
          </h2>
          <p className="muted">
            {detail.data.point_count.toLocaleString()} points · {fmtRange(detail.data.first_ts, detail.data.last_ts)} ·{' '}
            <span className={`badge badge-${detail.data.source}`}>{detail.data.source}</span>
          </p>
          {detail.data.description && <p className="muted">{detail.data.description}</p>}
        </div>
      )}

      {points.loading && <LoadingSkeleton height={400} />}
      {points.error && <ErrorState message={points.error.message} onRetry={points.retry} />}
      {points.data && points.data.count === 0 && (
        <EmptyState title="No data points" hint="This series exists but has no points stored yet." />
      )}
      {points.data && points.data.count > 0 && (
        <>
          {forecasts.data && forecasts.data.count > 0 && (
            <p className="muted small">
              Forecast provenance: {forecasts.data.forecasts[forecasts.data.count - 1].model_name} v
              {forecasts.data.forecasts[forecasts.data.count - 1].model_version} ·{' '}
              <StatusBadge
                label={`${forecasts.data.count} stored forecasts`}
                tone="info"
              />
            </p>
          )}
          {anomalies.data && anomalies.data.count > 0 && (
            <p className="muted small">
              Anomaly provenance: {anomalies.data.results[anomalies.data.count - 1].model_name} v
              {anomalies.data.results[anomalies.data.count - 1].model_version} ·{' '}
              {anomalies.data.results.filter((r) => r.is_anomaly).length} flagged of{' '}
              {anomalies.data.count} scored
            </p>
          )}
          <MetricChart
            points={points.data.points}
            forecasts={forecasts.data?.forecasts ?? []}
            anomalies={anomalies.data?.results ?? []}
            unit={detail.data?.unit}
          />
        </>
      )}
    </section>
  )
}
