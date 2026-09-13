import { useState } from 'react'
import { Overview } from './pages/Overview'
import { SeriesDetail } from './pages/SeriesDetail'

export default function App() {
  const [selectedId, setSelectedId] = useState<number | null>(null)

  return (
    <div className="app">
      <header className="app-header">
        <h1>PulseGuard</h1>
        <p className="muted">Metric anomaly detection &amp; probabilistic forecasting — Phase 1 preview</p>
      </header>
      <main className="app-main">
        {selectedId === null ? (
          <Overview onSelect={setSelectedId} />
        ) : (
          <SeriesDetail id={selectedId} onBack={() => setSelectedId(null)} />
        )}
      </main>
      <footer className="app-footer muted">
        Data comes live from the PulseGuard API — nothing on this page is mocked.
      </footer>
    </div>
  )
}
