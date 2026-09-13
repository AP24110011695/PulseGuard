import { api } from '../api/client'
import { EmptyState } from '../components/EmptyState'
import { ErrorState } from '../components/ErrorState'
import { LoadingSkeleton } from '../components/LoadingSkeleton'
import { useFetch } from '../hooks/useFetch'

export function Overview({ onSelect }: { onSelect: (id: number) => void }) {
  const { data, loading, error, retry } = useFetch(api.listSeries, [])

  if (loading) return <LoadingSkeleton height={200} />
  if (error) return <ErrorState message={error.message} onRetry={retry} />
  if (!data || data.length === 0) {
    return (
      <EmptyState
        title="No metric series yet"
        hint="Generate and ingest synthetic metrics with the simulator:"
        command="python -m ml.simulator --config ml/configs/simulator_default.yaml --ingest"
      />
    )
  }

  return (
    <section>
      <h2>Metric series</h2>
      <div className="card-grid">
        {data.map((s) => (
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
