/**
 * training_dashboard.jsx — Guardrail Arena Training Dashboard
 *
 * React dashboard for monitoring live training progress.
 * Polls the /training_log, /multi_agent_info, and /leaderboard endpoints.
 *
 * Usage:
 *   - Open in Claude Artifacts or any React environment
 *   - Set BASE_URL to your environment (default: HF Space)
 *   - The dashboard auto-refreshes every 5 seconds
 *
 * Dependencies: React, Recharts (CDN or npm)
 */

import React, { useState, useEffect } from "react";
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

// ── Config ────────────────────────────────────────────────────────────────────

let BASE_URL = "https://varunventra-guardrail-arena.hf.space";
const REFRESH_INTERVAL_MS = 5000;

const C = {
  // Backgrounds
  bg:      "#0b0f14",
  surface: "rgba(255,255,255,0.04)",
  surfaceHover: "rgba(255,255,255,0.07)",
  // Borders
  border:  "rgba(255,255,255,0.07)",
  // Text
  text:    "#e2e8f0",
  muted:   "#94a3b8",
  dim:     "#475569",
  // Brand
  blue:    "#3b82f6",
  blueD:   "#2563eb",
  green:   "#22c55e",
  red:     "#ef4444",
  orange:  "#f97316",
  purple:  "#a78bfa",
  yellow:  "#fbbf24",
  gray:    "#4b5563",
};

const TASK_LABELS = {
  basic_threat_detection: "Task 1: Basic Threat Detection",
  context_aware_policy:   "Task 2: Context-Aware Policy",
  multiturn_adversarial:  "Task 3: Multi-Turn Adversarial",
  adversarial_adaptation: "Task 4: Adversarial Adaptation",
};

// Real multi-task Colab results (Llama-3.1-8B zero-shot)
const BASELINES = {
  "All-Allow":   [0.3750, 0.4037, 0.1607, 0.1500],
  "All-Refuse":  [0.3534, 0.3460, 0.0688, 0.0000],
  "Llama-8B ZS": [0.6097, 0.5493, 0.3988, 0.0000],
  "GPT-4o-mini": [0.9216, 0.7512, 0.6120, 0.4820],
  "Qwen-235B":   [0.9857, 0.6862, 0.8275, 0.0000],
};

// ── Global CSS Injection ──────────────────────────────────────────────────────
// Provides :hover, :focus, :active, @keyframes — not possible with inline styles alone

function GlobalStyles() {
  return (
    <style>{`
      *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

      /* Cards */
      .ga-card {
        transition: box-shadow 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
        will-change: transform;
      }
      .ga-card:hover {
        box-shadow: 0 8px 32px rgba(0,0,0,0.5), 0 2px 8px rgba(0,0,0,0.3);
        transform: translateY(-1px);
        border-color: rgba(255,255,255,0.12) !important;
      }

      /* Buttons — primary */
      .ga-btn-primary {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border: 1px solid rgba(59,130,246,0.4);
        box-shadow: 0 2px 8px rgba(59,130,246,0.25);
        transition: all 0.15s ease;
        cursor: pointer;
      }
      .ga-btn-primary:hover {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        box-shadow: 0 4px 16px rgba(59,130,246,0.4);
        transform: scale(1.02);
      }
      .ga-btn-primary:active { transform: scale(0.97); }
      .ga-btn-primary:focus-visible {
        outline: none;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.35);
      }

      /* Buttons — purple */
      .ga-btn-purple {
        background: linear-gradient(135deg, #8b5cf6 0%, #7c3aed 100%);
        border: 1px solid rgba(139,92,246,0.4);
        box-shadow: 0 2px 8px rgba(139,92,246,0.25);
        transition: all 0.15s ease;
        cursor: pointer;
      }
      .ga-btn-purple:hover {
        background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%);
        box-shadow: 0 4px 16px rgba(139,92,246,0.4);
        transform: scale(1.02);
      }
      .ga-btn-purple:active { transform: scale(0.97); }
      .ga-btn-purple:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        transform: none;
      }
      .ga-btn-purple:focus-visible {
        outline: none;
        box-shadow: 0 0 0 3px rgba(139,92,246,0.35);
      }

      /* Ghost / secondary button */
      .ga-btn-ghost {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.1);
        transition: all 0.15s ease;
        cursor: pointer;
      }
      .ga-btn-ghost:hover {
        background: rgba(255,255,255,0.1);
        border-color: rgba(255,255,255,0.18);
        transform: scale(1.01);
      }
      .ga-btn-ghost:active { transform: scale(0.97); }

      /* Inputs */
      .ga-input {
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
        outline: none;
      }
      .ga-input:hover { border-color: rgba(255,255,255,0.18) !important; }
      .ga-input:focus {
        border-color: rgba(59,130,246,0.6) !important;
        box-shadow: 0 0 0 3px rgba(59,130,246,0.12);
      }

      /* Select */
      .ga-select {
        transition: border-color 0.15s ease;
        outline: none;
        cursor: pointer;
      }
      .ga-select:focus { border-color: rgba(59,130,246,0.5) !important; }

      /* Table rows */
      .ga-row { transition: background 0.12s ease; }
      .ga-row:hover { background: rgba(255,255,255,0.04) !important; }

      /* Info stat items */
      .ga-stat { transition: background 0.15s ease; }
      .ga-stat:hover { background: rgba(255,255,255,0.06) !important; }

      /* Skeleton shimmer */
      @keyframes ga-shimmer {
        0%   { background-position: -400px 0; }
        100% { background-position: 400px 0; }
      }
      .ga-skeleton {
        background: linear-gradient(90deg, rgba(255,255,255,0.04) 25%, rgba(255,255,255,0.09) 50%, rgba(255,255,255,0.04) 75%);
        background-size: 400px 100%;
        animation: ga-shimmer 1.6s ease-in-out infinite;
        border-radius: 6px;
      }

      /* Spinner */
      @keyframes ga-spin { to { transform: rotate(360deg); } }
      .ga-spinner { animation: ga-spin 0.8s linear infinite; }

      /* Pulse ring (status dot) */
      @keyframes ga-pulse { 0%, 100% { opacity: 0.5; transform: scale(1); } 50% { opacity: 0; transform: scale(2.2); } }
      .ga-pulse-ring { animation: ga-pulse 2s ease-in-out infinite; }

      /* Smooth number color transitions */
      .ga-metric { transition: color 0.3s ease; }

      /* Score bar fill */
      .ga-bar { transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1); }

      /* Link */
      .ga-link { transition: color 0.15s ease; text-decoration: none; }
      .ga-link:hover { color: #93c5fd !important; }
    `}</style>
  );
}

// ── API Helpers ───────────────────────────────────────────────────────────────

async function fetchJSON(url) {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(8000) });
    if (!res.ok) return null;
    return await res.json();
  } catch {
    return null;
  }
}

// ── Shared Primitives ─────────────────────────────────────────────────────────

function Spinner({ size = 20 }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "32px 0", gap: 10 }}>
      <svg className="ga-spinner" width={size} height={size} viewBox="0 0 20 20" fill="none">
        <circle cx="10" cy="10" r="8" stroke="rgba(255,255,255,0.1)" strokeWidth="2" />
        <path d="M10 2 A8 8 0 0 1 18 10" stroke={C.blue} strokeWidth="2.5" strokeLinecap="round" />
      </svg>
      <span style={{ color: C.muted, fontSize: 12 }}>Loading…</span>
    </div>
  );
}

function SkeletonLine({ width = "100%", height = 12, mb = 8 }) {
  return (
    <div
      className="ga-skeleton"
      style={{ width, height, marginBottom: mb, borderRadius: 6 }}
    />
  );
}

function SkeletonCard() {
  return (
    <div style={{ padding: 4 }}>
      <SkeletonLine width="40%" height={14} mb={16} />
      <SkeletonLine width="100%" height={10} mb={8} />
      <SkeletonLine width="85%" height={10} mb={8} />
      <SkeletonLine width="70%" height={10} mb={0} />
    </div>
  );
}

function EmptyState({ message, icon = "○" }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "40px 20px", gap: 10 }}>
      <span style={{ fontSize: 28, opacity: 0.2 }}>{icon}</span>
      <span style={{ color: C.muted, fontSize: 12, textAlign: "center", maxWidth: 220, lineHeight: 1.6 }}>{message}</span>
    </div>
  );
}

function ErrorState({ message, onRetry }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "36px 20px", gap: 10 }}>
      <span style={{ fontSize: 24, opacity: 0.6 }}>⚠</span>
      <span style={{ color: C.red, fontSize: 12, textAlign: "center", maxWidth: 260, lineHeight: 1.6 }}>{message}</span>
      {onRetry && (
        <button className="ga-btn-ghost" style={{ ...S.btn, marginTop: 4, fontSize: 11, padding: "4px 12px" }} onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

function Card({ children, style = {} }) {
  return (
    <div className="ga-card" style={{ ...S.card, ...style }}>
      {children}
    </div>
  );
}

function CardHeader({ title, right }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 14 }}>
      <span style={S.cardTitle}>{title}</span>
      {right}
    </div>
  );
}

function Badge({ children, color = C.blue, bg }) {
  return (
    <span style={{
      fontSize: 10,
      padding: "3px 9px",
      borderRadius: 20,
      background: bg || `${color}22`,
      color,
      border: `1px solid ${color}33`,
      fontWeight: 500,
      letterSpacing: "0.03em",
    }}>
      {children}
    </span>
  );
}

// ── Components ────────────────────────────────────────────────────────────────

// 1. Live Reward Curve
function RewardCurvePanel({ agentName }) {
  const [data, setData]           = useState([]);
  const [allAgents, setAllAgents] = useState([]);
  const [selected, setSelected]   = useState(agentName || "");
  const [loading, setLoading]     = useState(true);
  const [error, setError]         = useState(false);

  async function refresh(agent) {
    const all = await fetchJSON(`${BASE_URL}/training_log`);
    if (!all) { setError(true); setLoading(false); return; }
    if (all?.agents) setAllAgents(all.agents);
    const target = agent || all?.agents?.[0] || "";
    if (!target) { setLoading(false); return; }
    const log = await fetchJSON(`${BASE_URL}/training_log?agent_name=${encodeURIComponent(target)}`);
    setError(false);
    if (log?.entries) {
      setData(log.entries.map(e => ({ episode: e.episode, score: e.grader_score, reward: e.cumulative_reward })));
    }
    setLoading(false);
  }

  useEffect(() => {
    let cancelled = false;
    const run = () => !cancelled && refresh(selected);
    setLoading(true);
    run();
    const id = setInterval(run, REFRESH_INTERVAL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, [selected]);

  useEffect(() => { if (agentName) setSelected(agentName); }, [agentName]);

  const tooltipStyle = { backgroundColor: "#111827", border: `1px solid ${C.border}`, color: C.text, borderRadius: 8, fontSize: 12 };

  return (
    <Card>
      <CardHeader
        title="Live Reward Curve"
        right={
          <select className="ga-select" style={S.select} value={selected} onChange={e => setSelected(e.target.value)} aria-label="Select agent">
            <option value="">All agents</option>
            {allAgents.map(a => <option key={a} value={a}>{a}</option>)}
          </select>
        }
      />
      {loading ? <SkeletonCard /> : error ? <ErrorState message="Could not reach /training_log endpoint." onRetry={() => { setLoading(true); refresh(selected); }} /> : !data.length ? (
        <EmptyState icon="📈" message="No training data yet. Start a training run to see reward curves here." />
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data} margin={{ top: 4, right: 20, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="episode" stroke={C.dim} tick={{ fill: C.muted, fontSize: 11 }} />
            <YAxis domain={[0, 1]} stroke={C.dim} tick={{ fill: C.muted, fontSize: 11 }} />
            <Tooltip contentStyle={tooltipStyle} formatter={(v, n) => [typeof v === "number" ? v.toFixed(4) : v, n]} />
            <Legend wrapperStyle={{ color: C.muted, fontSize: 11 }} />
            <Line type="monotone" dataKey="score" stroke={C.blue} strokeWidth={2.5} dot={{ r: 3, fill: C.blue }} name="Grader Score" />
            {[
              { y: 0.3750, color: C.red,    label: "All-Allow" },
              { y: 0.6097, color: C.orange,  label: "Llama-8B ZS" },
            ].map(b => (
              <Line key={b.label} data={data.map(d => ({ ...d, [b.label]: b.y }))} dataKey={b.label}
                stroke={b.color} strokeDasharray="5 3" strokeWidth={1.5} dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )}
    </Card>
  );
}

// 2. Action Distribution
function ActionDistributionPanel({ sessionId }) {
  const [distData, setDistData] = useState([]);
  const [loading, setLoading]   = useState(false);
  const [error, setError]       = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    async function load() {
      setLoading(true);
      const bd = await fetchJSON(`${BASE_URL}/reward_breakdown?session_id=${sessionId}`);
      if (cancelled) return;
      if (!bd) { setError(true); setLoading(false); return; }
      const b = bd.breakdown || {};
      setError(false);
      setDistData([
        { name: "Allow",        value: b.correct_allows?.count  ?? 0, color: C.green  },
        { name: "Correct Block",value: b.correct_blocks?.count  ?? 0, color: C.blue   },
        { name: "Missed Attack",value: b.missed_attacks?.count  ?? 0, color: C.red    },
        { name: "Over-Block",   value: b.over_blocks?.count     ?? 0, color: C.orange },
      ].filter(d => d.value > 0));
      setLoading(false);
    }
    load();
    return () => { cancelled = true; };
  }, [sessionId]);

  const total = distData.reduce((s, d) => s + d.value, 0);
  const tooltipStyle = { backgroundColor: "#111827", border: `1px solid ${C.border}`, color: C.text, borderRadius: 8, fontSize: 12 };

  return (
    <Card>
      <CardHeader title="Action Distribution" right={<Badge color={C.blue}>Last Episode</Badge>} />
      {!sessionId ? (
        <EmptyState icon="🥧" message="Paste a session ID above to see action distribution." />
      ) : loading ? <SkeletonCard /> : error ? (
        <ErrorState message="Could not load reward breakdown." />
      ) : !distData.length ? (
        <EmptyState icon="○" message="No action data for this session." />
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie data={distData} cx="50%" cy="50%" innerRadius={55} outerRadius={85} paddingAngle={3} dataKey="value"
              label={({ name, value }) => `${name} (${((value / total) * 100).toFixed(0)}%)`}
              labelLine={{ stroke: C.dim }}>
              {distData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
            </Pie>
            <Tooltip contentStyle={tooltipStyle} />
          </PieChart>
        </ResponsiveContainer>
      )}
    </Card>
  );
}

// 3. Task Comparison — single fetch
function TaskComparisonPanel({ agentName }) {
  const tasks     = ["basic_threat_detection", "context_aware_policy", "multiturn_adversarial", "adversarial_adaptation"];
  const taskShort = ["Task 1", "Task 2", "Task 3", "Task 4"];
  const buildData = scores => tasks.map((t, i) => ({ task: taskShort[i], score: scores[i], allAllow: BASELINES["All-Allow"][i], llama8b: BASELINES["Llama-8B ZS"][i] }));

  const [data, setData]       = useState(buildData([null, null, null, null]));
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState(false);

  useEffect(() => {
    if (!agentName) return;
    let cancelled = false;
    async function load() {
      setLoading(true);
      const log = await fetchJSON(`${BASE_URL}/training_log?agent_name=${encodeURIComponent(agentName)}`);
      if (cancelled) return;
      if (!log) { setError(true); setLoading(false); return; }
      setError(false);
      const entries = log.entries ?? [];
      setData(buildData(tasks.map(t => {
        const te = entries.filter(e => e.task_id === t);
        return te.length ? te[te.length - 1].grader_score : null;
      })));
      setLoading(false);
    }
    load();
  }, [agentName]);

  const tooltipStyle = { backgroundColor: "#111827", border: `1px solid ${C.border}`, color: C.text, borderRadius: 8, fontSize: 12 };

  return (
    <Card>
      <CardHeader
        title="Task Comparison"
        right={agentName ? <Badge color={C.green}>{agentName}</Badge> : null}
      />
      {!agentName ? (
        <EmptyState icon="📊" message="Enter an agent name to compare performance across tasks." />
      ) : loading ? <SkeletonCard /> : error ? (
        <ErrorState message="Could not load agent training log." />
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={data} margin={{ top: 4, right: 20, left: 0, bottom: 4 }} barGap={4}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="task" stroke={C.dim} tick={{ fill: C.muted, fontSize: 11 }} />
            <YAxis domain={[0, 1]} stroke={C.dim} tick={{ fill: C.muted, fontSize: 11 }} />
            <Tooltip contentStyle={tooltipStyle} formatter={v => v != null ? v.toFixed(4) : "—"} />
            <Legend wrapperStyle={{ color: C.muted, fontSize: 11 }} />
            <Bar dataKey="score"    fill={C.blue}   name="Your Agent"  radius={[4, 4, 0, 0]} />
            <Bar dataKey="allAllow" fill={C.gray}   name="All-Allow"   radius={[4, 4, 0, 0]} />
            <Bar dataKey="llama8b"  fill={C.orange} name="Llama-8B ZS" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </Card>
  );
}

// 4. Task 4 Adversary Trajectory
function AdversaryStatePanel({ sessionId }) {
  const [trajectory, setTrajectory] = useState([]);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    async function load() {
      setLoading(true);
      const d = await fetchJSON(`${BASE_URL}/adversary_state?session_id=${sessionId}`);
      if (cancelled) return;
      if (!d) { setError(true); setLoading(false); return; }
      if (d?.adversary_trajectory) setTrajectory(d.adversary_trajectory);
      setLoading(false);
    }
    load();
    return () => { cancelled = true; };
  }, [sessionId]);

  const intensityColors = ["#22c55e", "#86efac", "#fbbf24", "#f97316", "#ef4444", "#a78bfa"];
  const intensityLabels = ["Safe", "Low", "Med", "Elev", "High", "Crit"];
  const topicsHit    = new Set(trajectory.map(t => t.topic_idx)).size;
  const correctCount = trajectory.filter(t => t.correct).length;
  const accuracy     = trajectory.length ? Math.round((correctCount / trajectory.length) * 100) : 0;
  const accColor     = accuracy >= 70 ? C.green : accuracy >= 40 ? C.orange : C.red;

  return (
    <Card>
      <CardHeader
        title="Task 4: Adversary Trajectory"
        right={topicsHit > 0 ? <Badge color={C.purple}>{topicsHit} topics</Badge> : null}
      />
      {!sessionId ? (
        <EmptyState icon="🎯" message="Run a Task 4 episode to see the adversary's decision trajectory." />
      ) : loading ? <SkeletonCard /> : error ? (
        <ErrorState message="Could not load adversary state." />
      ) : !trajectory.length ? (
        <EmptyState icon="○" message="No trajectory data yet." />
      ) : (
        <>
          {/* Summary metrics */}
          <div style={{ display: "flex", gap: 24, marginBottom: 14, paddingBottom: 14, borderBottom: `1px solid ${C.border}` }}>
            {[
              { label: "Turns",    value: trajectory.length, color: C.text   },
              { label: "Topics",   value: topicsHit,         color: C.purple },
              { label: "Accuracy", value: `${accuracy}%`,    color: accColor },
            ].map(s => (
              <div key={s.label}>
                <div className="ga-metric" style={{ fontSize: 22, fontWeight: 700, color: s.color, fontVariantNumeric: "tabular-nums" }}>{s.value}</div>
                <div style={{ fontSize: 10, color: C.muted, textTransform: "uppercase", letterSpacing: "0.06em", marginTop: 2 }}>{s.label}</div>
              </div>
            ))}
          </div>
          {/* Table */}
          <div style={{ overflowX: "auto", maxHeight: 220, overflowY: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
              <thead>
                <tr style={{ position: "sticky", top: 0, background: "#0f1117", zIndex: 1 }}>
                  {["Turn", "Topic", "Intensity", "Action", "✓"].map(h => (
                    <th key={h} style={{ padding: "7px 10px", color: C.dim, textAlign: "left", borderBottom: `1px solid ${C.border}`, fontWeight: 500, textTransform: "uppercase", fontSize: 10, letterSpacing: "0.06em" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {trajectory.map((step, i) => (
                  <tr key={i} className="ga-row" style={{ borderBottom: `1px solid rgba(255,255,255,0.04)` }}>
                    <td style={{ padding: "6px 10px", color: C.muted, fontVariantNumeric: "tabular-nums" }}>{step.turn}</td>
                    <td style={{ padding: "6px 10px", color: C.text }}>T{step.topic_idx}</td>
                    <td style={{ padding: "6px 10px" }}>
                      <span style={{ display: "inline-block", padding: "2px 7px", borderRadius: 20, background: intensityColors[step.intensity] + "20", color: intensityColors[step.intensity], fontSize: 10, fontWeight: 600, border: `1px solid ${intensityColors[step.intensity]}33` }}>
                        {intensityLabels[step.intensity]}
                      </span>
                    </td>
                    <td style={{ padding: "6px 10px" }}>
                      <span style={{ display: "inline-block", padding: "2px 7px", borderRadius: 20, background: step.agent_action === "allow" ? "#16a34a22" : "#dc262622", color: step.agent_action === "allow" ? C.green : C.red, fontSize: 10, border: `1px solid ${step.agent_action === "allow" ? "#16a34a44" : "#dc262644"}` }}>
                        {step.agent_action}
                      </span>
                    </td>
                    <td style={{ padding: "6px 10px", color: step.correct ? C.green : C.red, fontWeight: 700, fontSize: 14 }}>
                      {step.correct ? "✓" : "✗"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </Card>
  );
}

// 5. Leaderboard
function LeaderboardPanel() {
  const [entries, setEntries] = useState([]);
  const [task, setTask]       = useState("basic_threat_detection");
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(false);

  async function load() {
    const d = await fetchJSON(`${BASE_URL}/leaderboard`);
    if (!d) { setError(true); setLoading(false); return; }
    setError(false);
    setEntries(((d.leaderboard ?? []).filter(e => e.task_id === task)).slice(0, 10));
    setLoading(false);
  }

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    load();
    const id = setInterval(() => !cancelled && load(), REFRESH_INTERVAL_MS * 2);
    return () => { cancelled = true; clearInterval(id); };
  }, [task]);

  const maxScore = entries[0]?.score ?? 1;
  const rankColors = [C.yellow, "#9ca3af", "#cd7f32"];

  return (
    <Card>
      <CardHeader
        title="Leaderboard"
        right={
          <select className="ga-select" style={S.select} value={task} onChange={e => setTask(e.target.value)} aria-label="Select task">
            <option value="basic_threat_detection">Task 1 — Basic Threat</option>
            <option value="context_aware_policy">Task 2 — Context Policy</option>
            <option value="multiturn_adversarial">Task 3 — Multiturn Adv.</option>
            <option value="adversarial_adaptation">Task 4 — Adversarial Adapt.</option>
          </select>
        }
      />
      {loading ? <SkeletonCard /> : error ? (
        <ErrorState message="Could not load leaderboard." onRetry={() => { setLoading(true); load(); }} />
      ) : !entries.length ? (
        <EmptyState icon="🏆" message="No submissions yet for this task. Submit a score to appear here." />
      ) : (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr>
              {["Rank", "Agent", "Score"].map(h => (
                <th key={h} style={{ padding: "7px 10px", color: C.dim, textAlign: "left", borderBottom: `1px solid ${C.border}`, fontWeight: 500, textTransform: "uppercase", fontSize: 10, letterSpacing: "0.06em" }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {entries.map((e, i) => (
              <tr key={i} className="ga-row" style={{ borderBottom: `1px solid rgba(255,255,255,0.04)` }}>
                <td style={{ padding: "8px 10px", fontWeight: 700, color: rankColors[i] ?? C.dim, fontSize: 13 }}>
                  {i === 0 ? "🥇" : i === 1 ? "🥈" : i === 2 ? "🥉" : `#${i + 1}`}
                </td>
                <td style={{ padding: "8px 10px", color: i === 0 ? C.text : C.muted, fontWeight: i === 0 ? 600 : 400 }}>
                  {e.agent_name || "—"}
                </td>
                <td style={{ padding: "8px 10px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
                    <div style={{ flex: 1, maxWidth: 80, height: 5, background: "rgba(255,255,255,0.08)", borderRadius: 4, overflow: "hidden" }}>
                      <div className="ga-bar" style={{ height: "100%", width: `${((e.score ?? 0) / maxScore) * 100}%`, background: i === 0 ? C.yellow : C.blue, borderRadius: 4 }} />
                    </div>
                    <span className="ga-metric" style={{ color: i === 0 ? C.yellow : C.text, fontWeight: 700, fontVariantNumeric: "tabular-nums", minWidth: 48 }}>
                      {(e.score ?? 0).toFixed(4)}
                    </span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  );
}

// ── Task 4 Episode Runner ─────────────────────────────────────────────────────

async function runTask4Episode() {
  const resetRes = await fetchJSON(`${BASE_URL}/reset?task_id=adversarial_adaptation`);
  if (!resetRes?.session_id) throw new Error("Reset failed — is the environment online?");
  const sid = resetRes.session_id;
  let obs = resetRes, done = false, steps = 0;
  while (!done && steps < 35) {    // 30-turn episodes + margin
    const res = await fetch(`${BASE_URL}/step?session_id=${sid}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt_id: obs.prompt_id, action_type: obs.user_risk_score > 0.5 ? "refuse" : "allow", reason: "dashboard demo", modified_prompt: null }),
      signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) break;
    const d = await res.json();
    done = d.done;
    obs = d.observation || obs;
    steps++;
  }
  return { sessionId: sid, done, steps };
}

// ── Status Dot ────────────────────────────────────────────────────────────────

function StatusDot({ status }) {
  const color = status === "online" ? C.green : status === "offline" ? C.red : C.yellow;
  return (
    <div style={{ position: "relative", width: 10, height: 10, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
      {status === "online" && (
        <div className="ga-pulse-ring" style={{ position: "absolute", inset: -3, borderRadius: "50%", background: C.green, opacity: 0.4 }} />
      )}
      <div style={{ width: 8, height: 8, borderRadius: "50%", background: color, position: "relative", boxShadow: `0 0 6px ${color}66` }} />
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────

export default function TrainingDashboard() {
  const [urlInput, setUrlInput]             = useState(BASE_URL);
  const [agentName, setAgentName]           = useState("guardrail_trl_agent");
  const [sessionId, setSessionId]           = useState("");
  const [task4SessionId, setTask4SessionId] = useState("");
  const [envStatus, setEnvStatus]           = useState("checking");
  const [lastRefresh, setLastRefresh]       = useState(new Date());
  const [task4Running, setTask4Running]     = useState(false);
  const [task4Msg, setTask4Msg]             = useState("");

  function handleConnect() {
    BASE_URL = urlInput.replace(/\/+$/, "");
    setEnvStatus("checking");
  }

  async function handleRunTask4() {
    setTask4Running(true);
    setTask4Msg("Running episode…");
    setTask4SessionId("");
    try {
      const { sessionId: sid, done, steps } = await runTask4Episode();
      setTask4SessionId(sid);
      setTask4Msg(done ? `Done · ${steps} steps · ${sid.slice(0, 8)}…` : `Partial (${steps} steps) · ${sid.slice(0, 8)}…`);
    } catch (e) {
      setTask4Msg(`Error: ${e.message}`);
    } finally {
      setTask4Running(false);
    }
  }

  useEffect(() => {
    let cancelled = false;
    async function checkHealth() {
      const r = await fetchJSON(`${BASE_URL}/health`);
      if (!cancelled) { setEnvStatus(r ? "online" : "offline"); setLastRefresh(new Date()); }
    }
    checkHealth();
    const id = setInterval(checkHealth, REFRESH_INTERVAL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  const statusText = { online: "Online", offline: "Offline", checking: "Checking…" };

  return (
    <>
      <GlobalStyles />
      <div style={S.root}>

        {/* ── Header ── */}
        <header style={S.header}>
          <div>
            <h1 style={S.title}>Guardrail Arena</h1>
            <p style={S.subtitle}>Training Dashboard · Multi-Agent Interactions</p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <StatusDot status={envStatus} />
            <span style={{ color: envStatus === "offline" ? C.red : C.muted, fontSize: 12 }}>
              {statusText[envStatus]}
            </span>
            <span style={{ color: C.dim, fontSize: 11 }}>·</span>
            <span style={{ color: C.dim, fontSize: 11 }}>{lastRefresh.toLocaleTimeString()}</span>
          </div>
        </header>

        {/* ── URL Bar ── */}
        <div style={S.urlBar}>
          <span style={{ color: C.dim, fontSize: 11, whiteSpace: "nowrap" }}>ENV</span>
          <input
            className="ga-input"
            style={{ ...S.input, flex: 1 }}
            value={urlInput}
            onChange={e => setUrlInput(e.target.value)}
            onKeyDown={e => e.key === "Enter" && handleConnect()}
            placeholder="https://varunventra-guardrail-arena.hf.space"
            aria-label="Environment URL"
          />
          <button className="ga-btn-primary" style={S.btn} onClick={handleConnect}>Connect</button>
        </div>

        {/* ── Key Numbers Strip ── */}
        <div style={S.infoBar}>
          {[
            { label: "Qwen-235B · Task 4",    value: "0.0000", color: C.red    },
            { label: "Q-learner · Task 4",     value: "0.9540", color: C.green  },
            { label: "Llama-8B ZS · Task 1",  value: "0.6097", color: C.muted  },
            { label: "Llama-8B SFT · Task 1", value: "0.9504", color: C.blue   },
            { label: "Branching Convos",        value: "30",     color: C.text   },
            { label: "Adversary States",        value: "180",    color: C.text   },
          ].map((item, i, arr) => (
            <div key={item.label} className="ga-stat" style={{ ...S.infoItem, borderRight: i < arr.length - 1 ? `1px solid ${C.border}` : "none" }}>
              <span style={S.infoLabel}>{item.label}</span>
              <span className="ga-metric" style={{ ...S.infoValue, color: item.color }}>{item.value}</span>
            </div>
          ))}
        </div>

        {/* ── Multi-Agent Architecture ── */}
        <Card style={{ marginBottom: 16 }}>
          <CardHeader
            title="Multi-Agent Architecture"
            right={<Badge color={C.purple}>Theme #1: Multi-Agent Interactions</Badge>}
          />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div style={S.agentBox}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                <span style={{ fontSize: 16 }}>⚔</span>
                <span style={{ color: C.red, fontWeight: 700, fontSize: 13 }}>Adversary: DeterministicAdversary FSM</span>
              </div>
              <div style={{ color: C.muted, fontSize: 12, lineHeight: 1.75 }}>
                60 states (10 topics × 6 intensities) × 3 variants = <strong style={{ color: C.text }}>180 observable states</strong><br />
                Adapts in real-time to defender actions<br />
                <span style={{ color: C.orange }}>allow → escalate · block → back off</span>
              </div>
            </div>
            <div style={S.agentBox}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 10 }}>
                <span style={{ fontSize: 16 }}>🛡</span>
                <span style={{ color: C.green, fontWeight: 700, fontSize: 13 }}>Defender: Trainable Safety Policy</span>
              </div>
              <div style={{ color: C.muted, fontSize: 12, lineHeight: 1.75 }}>
                Observation: <strong style={{ color: C.text }}>prompt + context + risk_score</strong><br />
                Actions: allow / refuse / modify / escalate<br />
                <span style={{ color: C.blue }}>+0.20 correct block · −0.30 missed attack</span>
              </div>
            </div>
          </div>
        </Card>

        {/* ── Controls ── */}
        <div style={S.controls}>
          <div style={S.inputGroup}>
            <label htmlFor="agentNameInput" style={S.label}>Agent Name</label>
            <input
              id="agentNameInput"
              className="ga-input"
              style={S.input}
              value={agentName}
              onChange={e => setAgentName(e.target.value)}
              placeholder="e.g. guardrail_trl_agent"
            />
          </div>
          <div style={S.inputGroup}>
            <label htmlFor="sessionInput" style={S.label}>
              Session ID <span style={{ color: C.dim, fontWeight: 400, textTransform: "none" }}>for action breakdown</span>
            </label>
            <input
              id="sessionInput"
              className="ga-input"
              style={S.input}
              value={sessionId}
              onChange={e => setSessionId(e.target.value)}
              placeholder="Paste UUID from /reset"
            />
          </div>
          <div style={S.inputGroup}>
            <label style={S.label}>Task 4 Demo</label>
            <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
              <button
                className="ga-btn-purple"
                style={{ ...S.btn, minWidth: 160 }}
                onClick={handleRunTask4}
                disabled={task4Running}
                aria-busy={task4Running}
              >
                {task4Running ? (
                  <span style={{ display: "flex", alignItems: "center", gap: 7 }}>
                    <svg className="ga-spinner" width="13" height="13" viewBox="0 0 13 13" fill="none">
                      <circle cx="6.5" cy="6.5" r="5" stroke="rgba(255,255,255,0.2)" strokeWidth="1.5" />
                      <path d="M6.5 1.5 A5 5 0 0 1 11.5 6.5" stroke="white" strokeWidth="1.5" strokeLinecap="round" />
                    </svg>
                    Running…
                  </span>
                ) : "▶  Run Task 4 Episode"}
              </button>
              {task4Msg && (
                <span style={{ fontSize: 11, color: task4Msg.startsWith("Error") ? C.red : C.muted, maxWidth: 200 }}>
                  {task4Msg}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* ── Main 2×2 Grid ── */}
        <div style={S.grid}>
          <RewardCurvePanel agentName={agentName} />
          <ActionDistributionPanel sessionId={sessionId} />
          <TaskComparisonPanel agentName={agentName} />
          <AdversaryStatePanel sessionId={task4SessionId} />
        </div>

        {/* ── Leaderboard (full width) ── */}
        <div style={{ marginTop: 16 }}>
          <LeaderboardPanel />
        </div>

        {/* ── Footer ── */}
        <footer style={S.footer}>
          <span>Guardrail Arena · Meta × HuggingFace × Cerebral Valley OpenEnv Hackathon</span>
          <div style={{ display: "flex", gap: 20 }}>
            <a className="ga-link" href={`${BASE_URL}/docs`} target="_blank" rel="noopener noreferrer" style={{ color: C.blue }}>API Docs</a>
            <a className="ga-link" href="https://github.com/sahithsundarw/sentinel" target="_blank" rel="noopener noreferrer" style={{ color: C.muted }}>GitHub</a>
          </div>
        </footer>
      </div>
    </>
  );
}

// ── Style Constants ───────────────────────────────────────────────────────────

const S = {
  root: {
    backgroundColor: C.bg,
    minHeight: "100vh",
    fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
    color: C.text,
    padding: "24px",
    maxWidth: 1200,
    margin: "0 auto",
    WebkitFontSmoothing: "antialiased",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
    paddingBottom: 16,
    borderBottom: `1px solid ${C.border}`,
  },
  title: {
    margin: 0,
    fontSize: 22,
    fontWeight: 700,
    color: C.text,
    letterSpacing: "-0.4px",
  },
  subtitle: {
    margin: "5px 0 0",
    fontSize: 13,
    color: C.muted,
    letterSpacing: "0.01em",
  },
  urlBar: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    padding: "8px 12px",
    background: C.surface,
    border: `1px solid ${C.border}`,
    borderRadius: 10,
    marginBottom: 14,
  },
  infoBar: {
    display: "flex",
    background: C.surface,
    border: `1px solid ${C.border}`,
    borderRadius: 12,
    marginBottom: 16,
    overflow: "hidden",
  },
  infoItem: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
    padding: "12px 20px",
    flex: 1,
  },
  infoLabel: {
    fontSize: 10,
    color: C.dim,
    textTransform: "uppercase",
    letterSpacing: "0.07em",
    whiteSpace: "nowrap",
  },
  infoValue: {
    fontSize: 20,
    fontWeight: 700,
    fontVariantNumeric: "tabular-nums",
  },
  controls: {
    display: "flex",
    gap: 16,
    marginBottom: 16,
    flexWrap: "wrap",
    alignItems: "flex-end",
  },
  inputGroup: {
    display: "flex",
    flexDirection: "column",
    gap: 6,
  },
  label: {
    fontSize: 11,
    color: C.muted,
    textTransform: "uppercase",
    letterSpacing: "0.07em",
    fontWeight: 600,
  },
  input: {
    background: C.surface,
    border: `1px solid ${C.border}`,
    borderRadius: 8,
    color: C.text,
    padding: "8px 12px",
    fontSize: 13,
    minWidth: 240,
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(2, 1fr)",
    gap: 16,
  },
  card: {
    background: C.surface,
    border: `1px solid ${C.border}`,
    borderRadius: 14,
    padding: "18px 20px",
  },
  agentBox: {
    background: "rgba(0,0,0,0.25)",
    borderRadius: 10,
    padding: "14px 18px",
    border: `1px solid ${C.border}`,
  },
  cardTitle: {
    fontSize: 14,
    fontWeight: 600,
    color: C.text,
    letterSpacing: "-0.1px",
  },
  select: {
    background: C.surface,
    border: `1px solid ${C.border}`,
    borderRadius: 6,
    color: C.text,
    padding: "4px 8px",
    fontSize: 11,
  },
  btn: {
    border: "none",
    borderRadius: 8,
    color: "#fff",
    padding: "8px 16px",
    fontSize: 12,
    fontWeight: 600,
    whiteSpace: "nowrap",
    display: "inline-flex",
    alignItems: "center",
    justifyContent: "center",
    gap: 6,
  },
  footer: {
    marginTop: 24,
    paddingTop: 16,
    borderTop: `1px solid ${C.border}`,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    fontSize: 11,
    color: C.dim,
  },
};
