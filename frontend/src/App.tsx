import { useState } from 'react'
import { Alerts } from './pages/Alerts'
import { Drift } from './pages/Drift'
import { Models } from './pages/Models'
import { Overview } from './pages/Overview'
import { SeriesDetail } from './pages/SeriesDetail'

type View = { page: 'overview' } | { page: 'series'; id: number } | { page: 'models' } | { page: 'drift' } | { page: 'alerts' }

const NAV: { key: View['page']; label: string }[] = [
  { key: 'overview', label: 'Overview' },
  { key: 'models', label: 'Models' },
  { key: 'drift', label: 'Drift' },
  { key: 'alerts', label: 'Alerts' },
]

export default function App() {
  const [view, setView] = useState<View>({ page: 'overview' })

  return (
    <div className="app">
      <header className="app-header">
        <h1>PulseGuard</h1>
        <p className="muted">Metric anomaly detection &amp; probabilistic forecasting</p>
        <nav className="app-nav" aria-label="Main navigation">
          {NAV.map((item) => (
            <button
              key={item.key}
              type="button"
              className={view.page === item.key ? 'nav-link active' : 'nav-link'}
              onClick={() => setView({ page: item.key } as View)}
            >
              {item.label}
            </button>
          ))}
        </nav>
      </header>
      <main className="app-main">
        {view.page === 'overview' && <Overview onSelect={(id) => setView({ page: 'series', id })} />}
        {view.page === 'series' && <SeriesDetail id={view.id} onBack={() => setView({ page: 'overview' })} />}
        {view.page === 'models' && <Models />}
        {view.page === 'drift' && <Drift />}
        {view.page === 'alerts' && <Alerts />}
      </main>
      <footer className="app-footer muted">
        Data comes live from the PulseGuard API — nothing on this page is mocked.
      </footer>
    </div>
  )
}
