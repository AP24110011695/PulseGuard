// 127.0.0.1 (not "localhost"): on dual-stack Windows setups another WSL2-relayed
// service can own localhost's IPv6 loopback; the IPv4 loopback is unambiguous.
const API_BASE: string = import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000'

export interface MetricSeries {
  id: number
  name: string
  unit: string | null
  description: string | null
  source: string
  tags: Record<string, string> | null
  created_at: string
}

export interface MetricSeriesDetail extends MetricSeries {
  point_count: number
  first_ts: string | null
  last_ts: string | null
}

export interface Point {
  ts: string
  value: number
  source_label: boolean | null
}

export interface PointsPage {
  series_id: number
  start: string | null
  end: string | null
  max_points: number
  downsampled: boolean
  bucket_seconds: number | null
  count: number
  points: Point[]
}

export class ApiError extends Error {
  readonly status: number

  constructor(message: string, status: number) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${path}`)
  } catch {
    throw new ApiError(`Cannot reach the PulseGuard API at ${API_BASE}`, 0)
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      if (typeof body.detail === 'string') detail = body.detail
    } catch {
      // keep the status text when the error body is not the standard shape
    }
    throw new ApiError(`API error ${res.status}: ${detail}`, res.status)
  }
  return res.json() as Promise<T>
}

export const api = {
  listSeries: () => request<MetricSeries[]>('/api/v1/series'),
  getSeries: (id: number) => request<MetricSeriesDetail>(`/api/v1/series/${id}`),
  getPoints: (id: number, maxPoints = 2000) =>
    request<PointsPage>(`/api/v1/series/${id}/points?max_points=${maxPoints}`),
}
