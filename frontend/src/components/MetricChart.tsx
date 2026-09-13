import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { Point } from '../api/client'

function formatTick(ts: string): string {
  return ts.slice(0, 16).replace('T', ' ')
}

export function MetricChart({ points, unit }: { points: Point[]; unit?: string | null }) {
  const data = points.map((p) => ({ ts: p.ts, value: p.value }))
  return (
    <div className="chart-frame">
      <ResponsiveContainer width="100%" height={360}>
        <LineChart data={data} margin={{ top: 12, right: 24, bottom: 8, left: 8 }}>
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
            formatter={(value: number | string) => [
              typeof value === 'number' ? value.toFixed(3) : value,
              unit ?? 'value',
            ]}
          />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#4f46e5"
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
