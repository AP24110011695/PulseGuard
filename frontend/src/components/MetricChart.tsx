import {
  Area,
  Brush,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { AnomalyResult, Point, StoredForecast } from '../api/client'

interface ChartRow {
  ts: string
  value?: number
  p05?: number
  p50?: number
  p95?: number
  anomalyValue?: number
}

function formatTick(ts: string): string {
  return ts.slice(0, 16).replace('T', ' ')
}

export function MetricChart({
  points,
  forecasts = [],
  anomalies = [],
  unit,
}: {
  points: Point[]
  forecasts?: StoredForecast[]
  anomalies?: AnomalyResult[]
  unit?: string | null
}) {
  const byTs = new Map<string, ChartRow>()
  for (const p of points) byTs.set(p.ts, { ts: p.ts, value: p.value })
  for (const f of forecasts) {
    const row = byTs.get(f.target_ts) ?? { ts: f.target_ts }
    row.p05 = f.quantiles['0.05']
    row.p50 = f.quantiles['0.5']
    row.p95 = f.quantiles['0.95']
    byTs.set(f.target_ts, row)
  }
  for (const a of anomalies) {
    if (!a.is_anomaly) continue
    const row = byTs.get(a.ts)
    if (row) row.anomalyValue = row.value
  }
  const data = [...byTs.values()].sort((a, b) => a.ts.localeCompare(b.ts))

  return (
    <div className="chart-frame">
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={data} margin={{ top: 12, right: 24, bottom: 8, left: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
          <XAxis
            dataKey="ts"
            tickFormatter={formatTick}
            minTickGap={80}
            tick={{ fontSize: 11 }}
            stroke="#6b7280"
          />
          <YAxis
            tick={{ fontSize: 11 }}
            stroke="#6b7280"
            width={56}
            label={
              unit
                ? { value: unit, angle: -90, position: 'insideLeft', fontSize: 11, fill: '#6b7280' }
                : undefined
            }
          />
          <Tooltip
            labelFormatter={formatTick}
            formatter={(value: number | string | (number | string)[]) => {
              if (Array.isArray(value)) return [`${value[0]} – ${value[1]}`, 'band']
              return [typeof value === 'number' ? value.toFixed(3) : value, 'value']
            }}
          />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {forecasts.length > 0 && (
            <Area
              dataKey={(row: ChartRow) =>
                row.p05 !== undefined && row.p95 !== undefined ? [row.p05, row.p95] : [null, null]
              }
              name="forecast band (p05–p95)"
              stroke="none"
              fill="#4f46e5"
              fillOpacity={0.12}
              isAnimationActive={false}
              connectNulls
            />
          )}
          <Line
            type="monotone"
            dataKey="value"
            name={`value${unit ? ` (${unit})` : ''}`}
            stroke="#4f46e5"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          {forecasts.length > 0 && (
            <Line
              type="monotone"
              dataKey="p50"
              name="forecast median"
              stroke="#f59e0b"
              strokeWidth={1.2}
              strokeDasharray="4 3"
              dot={false}
              isAnimationActive={false}
            />
          )}
          {anomalies.length > 0 && (
            <Scatter
              dataKey="anomalyValue"
              name="flagged anomaly"
              fill="#dc2626"
              shape="circle"
              isAnimationActive={false}
            />
          )}
          {data.length > 300 && <Brush dataKey="ts" height={22} stroke="#4f46e5" />}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}
