/**
 * training_dashboard.jsx — Sentinel Training Dashboard
 * Dependencies: React, Recharts
 */

import React, { useState, useEffect } from "react";
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

// ── Config ────────────────────────────────────────────────────────────────────

let BASE_URL = "https://varunventra-guardrail-arena.hf.space";
const REFRESH_MS = 5000;

const C = {
  bg:       "#070b10",
  surface:  "rgba(255,255,255,0.055)",
  surfaceM: "rgba(255,255,255,0.085)",
  border:   "rgba(255,255,255,0.09)",
  borderB:  "rgba(255,255,255,0.14)",
  text:     "#f1f5f9",
  sub:      "#cbd5e1",
  muted:    "#64748b",
  dim:      "#334155",
  blue:     "#3b82f6",
  blueL:    "#60a5fa",
  green:    "#22c55e",
  greenL:   "#4ade80",
  red:      "#ef4444",
  redL:     "#f87171",
  orange:   "#f97316",
  purple:   "#a78bfa",
  purpleD:  "#7c3aed",
  yellow:   "#fbbf24",
  gray:     "#475569",
};

const BASELINES = {
  "All-Allow":   [0.3750, 0.4037, 0.1607, 0.1500],
  "Llama-8B ZS": [0.6097, 0.5493, 0.3988, 0.0000],
  "GPT-4o-mini": [0.9216, 0.7512, 0.6120, 0.4820],
  "Qwen-235B":   [0.9857, 0.6862, 0.8275, 0.0000],
};

// ── Global CSS ────────────────────────────────────────────────────────────────

function GlobalStyles() {
  return (
    <style>{`
      *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
      ::-webkit-scrollbar { width: 5px; height: 5px; }
      ::-webkit-scrollbar-track { background: transparent; }
      ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 4px; }

      /* ── Page bg gradient ── */
      .ga-root {
        background:
          radial-gradient(ellipse 60% 40% at 10% 0%, rgba(59,130,246,0.10) 0%, transparent 70%),
          radial-gradient(ellipse 40% 30% at 90% 100%, rgba(139,92,246,0.08) 0%, transparent 60%),
          #070b10;
      }

      /* ── Cards ── */
      .ga-card {
        background: linear-gradient(145deg, rgba(255,255,255,0.07) 0%, rgba(255,255,255,0.03) 100%);
        border: 1px solid rgba(255,255,255,0.09);
        border-radius: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
        transition: box-shadow 0.25s ease, border-color 0.25s ease, transform 0.2s ease;
        will-change: transform;
      }
      .ga-card:hover {
        box-shadow: 0 8px 40px rgba(0,0,0,0.55), 0 2px 8px rgba(0,0,0,0.3), inset 0 1px 0 rgba(255,255,255,0.1);
        border-color: rgba(255,255,255,0.16);
        transform: translateY(-2px);
      }

      /* ── Stat tiles (colored accent) ── */
      .ga-stat-tile {
        transition: background 0.2s ease, transform 0.15s ease;
        cursor: default;
      }
      .ga-stat-tile:hover {
        background: rgba(255,255,255,0.07) !important;
        transform: translateY(-1px);
      }

      /* ── Buttons ── */
      .ga-btn-blue {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
        border: 1px solid rgba(59,130,246,0.5);
        box-shadow: 0 2px 8px rgba(59,130,246,0.3), inset 0 1px 0 rgba(255,255,255,0.15);
        transition: all 0.18s ease;
        cursor: pointer;
        color: #fff;
      }
      .ga-btn-blue:hover {
        background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%);
        box-shadow: 0 0 24px rgba(59,130,246,0.45), 0 4px 12px rgba(59,130,246,0.3), inset 0 1px 0 rgba(255,255,255,0.2);
        transform: scale(1.025);
      }
      .ga-btn-blue:active { transform: scale(0.97); box-shadow: 0 1px 4px rgba(59,130,246,0.3); }

      /* ── Primary CTA (purple, glowing) ── */
      .ga-cta {
        background: linear-gradient(135deg, #8b5cf6 0%, #6d28d9 100%);
        border: 1px solid rgba(139,92,246,0.6);
        box-shadow: 0 0 0 0 rgba(139,92,246,0.4);
        transition: all 0.2s ease;
        cursor: pointer;
        color: #fff;
        animation: ga-cta-idle 3s ease-in-out infinite;
      }
      @keyframes ga-cta-idle {
        0%, 100% { box-shadow: 0 0 12px rgba(139,92,246,0.3), 0 4px 16px rgba(139,92,246,0.2), inset 0 1px 0 rgba(255,255,255,0.15); }
        50%       { box-shadow: 0 0 24px rgba(139,92,246,0.5), 0 4px 24px rgba(139,92,246,0.3), inset 0 1px 0 rgba(255,255,255,0.2); }
      }
      .ga-cta:hover {
        background: linear-gradient(135deg, #a78bfa 0%, #8b5cf6 100%);
        box-shadow: 0 0 32px rgba(139,92,246,0.6), 0 8px 24px rgba(139,92,246,0.35), inset 0 1px 0 rgba(255,255,255,0.25);
        transform: scale(1.03);
        animation: none;
      }
      .ga-cta:active { transform: scale(0.96); animation: none; }
      .ga-cta:disabled { opacity: 0.55; cursor: not-allowed; transform: none; animation: none; }

      /* ── Ghost / secondary ── */
      .ga-btn-ghost {
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.12);
        transition: all 0.15s ease;
        cursor: pointer;
        color: ${C.sub};
      }
      .ga-btn-ghost:hover {
        background: rgba(255,255,255,0.12);
        border-color: rgba(255,255,255,0.2);
        transform: scale(1.01);
      }
      .ga-btn-ghost:active { transform: scale(0.97); }

      /* ── Inputs ── */
      .ga-input {
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.09);
        color: ${C.text};
        transition: border-color 0.15s ease, box-shadow 0.15s ease;
        outline: none;
      }
      .ga-input:hover { border-color: rgba(255,255,255,0.16); }
      .ga-input:focus {
        border-color: rgba(59,130,246,0.7);
        box-shadow: 0 0 0 3px rgba(59,130,246,0.14), 0 0 12px rgba(59,130,246,0.1);
      }

      /* ── Select ── */
      .ga-select {
        background: rgba(0,0,0,0.3);
        border: 1px solid rgba(255,255,255,0.09);
        color: ${C.sub};
        outline: none;
        cursor: pointer;
        transition: border-color 0.15s ease;
      }
      .ga-select:hover { border-color: rgba(255,255,255,0.18); }
      .ga-select:focus { border-color: rgba(59,130,246,0.6); }

      /* ── Table rows ── */
      .ga-row { transition: background 0.1s ease; }
      .ga-row:hover { background: rgba(255,255,255,0.045) !important; }

      /* ── Skeleton shimmer ── */
      @keyframes ga-shimmer {
        0%   { background-position: -600px 0; }
        100% { background-position: 600px 0; }
      }
      .ga-skeleton {
        background: linear-gradient(90deg,
          rgba(255,255,255,0.04) 0%,
          rgba(255,255,255,0.08) 50%,
          rgba(255,255,255,0.04) 100%);
        background-size: 600px 100%;
        animation: ga-shimmer 1.8s ease-in-out infinite;
        border-radius: 6px;
      }

      /* ── Spinner ── */
      @keyframes ga-spin { to { transform: rotate(360deg); } }
      .ga-spin { animation: ga-spin 0.75s linear infinite; }

      /* ── Status pulse ── */
      @keyframes ga-pulse { 0%, 100% { opacity: 0.6; transform: scale(1); } 50% { opacity: 0; transform: scale(2.8); } }
      .ga-pulse { animation: ga-pulse 2.2s ease-in-out infinite; }

      /* ── Gradient title text ── */
      .ga-title-gradient {
        background: linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
      }

      /* ── Score bar fill ── */
      .ga-fill { transition: width 0.5s cubic-bezier(0.4,0,0.2,1); }

      /* ── Links ── */
      .ga-link { transition: color 0.15s ease; text-decoration: none; }
      .ga-link:hover { color: #93c5fd !important; }

      /* ── Divider line ── */
      .ga-divider { border: none; border-top: 1px solid rgba(255,255,255,0.07); margin: 14px 0; }
    `}</style>
  );
}

// ── API ───────────────────────────────────────────────────────────────────────

async function fetchJSON(url) {
  try {
    const r = await fetch(url, { signal: AbortSignal.timeout(8000) });
    if (!r.ok) return null;
    return r.json();
  } catch { return null; }
}

// ── Primitives ────────────────────────────────────────────────────────────────

function Spinner({ size = 18 }) {
  return (
    <svg className="ga-spin" width={size} height={size} viewBox="0 0 18 18" fill="none">
      <circle cx="9" cy="9" r="7" stroke="rgba(255,255,255,0.1)" strokeWidth="2" />
      <path d="M9 2 A7 7 0 0 1 16 9" stroke={C.blue} strokeWidth="2.5" strokeLinecap="round" />
    </svg>
  );
}

function SkeletonBlock({ w = "100%", h = 12, mb = 10 }) {
  return <div className="ga-skeleton" style={{ width: w, height: h, marginBottom: mb }} />;
}

function SkeletonCard() {
  return (
    <div style={{ padding: "4px 0" }}>
      <SkeletonBlock w="45%" h={14} mb={18} />
      <SkeletonBlock w="100%" h={9} mb={9} />
      <SkeletonBlock w="88%" h={9} mb={9} />
      <SkeletonBlock w="72%" h={9} mb={0} />
    </div>
  );
}

function EmptyState({ icon = "○", msg }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10, padding: "44px 20px" }}>
      <span style={{ fontSize: 30, opacity: 0.18, filter: "grayscale(1)" }}>{icon}</span>
      <p style={{ color: C.muted, fontSize: 12.5, textAlign: "center", maxWidth: 220, lineHeight: 1.65, margin: 0 }}>{msg}</p>
    </div>
  );
}

function ErrorState({ msg, onRetry }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 10, padding: "40px 20px" }}>
      <span style={{ fontSize: 22, color: C.red, opacity: 0.7 }}>⚠</span>
      <p style={{ color: C.red, fontSize: 12, textAlign: "center", maxWidth: 260, lineHeight: 1.6, margin: 0, opacity: 0.85 }}>{msg}</p>
      {onRetry && (
        <button className="ga-btn-ghost" style={{ ...BTN, padding: "5px 14px", fontSize: 11, marginTop: 4 }} onClick={onRetry}>
          Retry
        </button>
      )}
    </div>
  );
}

function Card({ children, style = {}, accent }) {
  const accentStyle = accent
    ? { borderTop: `2px solid ${accent}`, borderRadius: "16px" }
    : {};
  return (
    <div className="ga-card" style={{ padding: "20px 22px", ...accentStyle, ...style }}>
      {children}
    </div>
  );
}

function CardHead({ title, right }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 16 }}>
      <span style={{ fontSize: 13.5, fontWeight: 600, color: C.text, letterSpacing: "-0.1px" }}>{title}</span>
      {right}
    </div>
  );
}

function Pill({ children, color = C.blue }) {
  return (
    <span style={{
      fontSize: 10, fontWeight: 600, padding: "3px 10px", borderRadius: 20,
      background: `${color}1a`, color, border: `1px solid ${color}35`,
      letterSpacing: "0.04em", whiteSpace: "nowrap",
    }}>
      {children}
    </span>
  );
}

// Shared tooltip style for recharts
const TIP = {
  contentStyle: {
    backgroundColor: "#111827",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 10,
    color: C.text,
    fontSize: 12,
    boxShadow: "0 8px 32px rgba(0,0,0,0.5)",
  },
};

// ── Shared button shape (class provides colors) ───────────────────────────────
const BTN = {
  border: "none",
  borderRadius: 9,
  padding: "9px 18px",
  fontSize: 12.5,
  fontWeight: 700,
  display: "inline-flex",
  alignItems: "center",
  justifyContent: "center",
  gap: 7,
  whiteSpace: "nowrap",
  letterSpacing: "0.01em",
};

// ── Components ────────────────────────────────────────────────────────────────

function RewardCurvePanel({ agentName }) {
  const [data, setData]         = useState([]);
  const [agents, setAgents]     = useState([]);
  const [agent, setAgent]       = useState(agentName || "");
  const [loading, setLoading]   = useState(true);
  const [error, setError]       = useState(false);

  async function refresh(a) {
    const all = await fetchJSON(`${BASE_URL}/training_log`);
    if (!all) { setError(true); setLoading(false); return; }
    if (all.agents) setAgents(all.agents);
    const target = a || all.agents?.[0] || "";
    if (!target) { setLoading(false); return; }
    const log = await fetchJSON(`${BASE_URL}/training_log?agent_name=${encodeURIComponent(target)}`);
    setError(false);
    if (log?.entries) setData(log.entries.map(e => ({ ep: e.episode, score: e.grader_score })));
    setLoading(false);
  }

  useEffect(() => {
    let dead = false;
    setLoading(true); setError(false);
    refresh(agent);
    const id = setInterval(() => !dead && refresh(agent), REFRESH_MS);
    return () => { dead = true; clearInterval(id); };
  }, [agent]);

  useEffect(() => { if (agentName) setAgent(agentName); }, [agentName]);

  return (
    <Card accent={C.blue}>
      <CardHead
        title="Live Reward Curve"
        right={
          <select className="ga-select" style={{ ...S.select }} value={agent}
            onChange={e => setAgent(e.target.value)} aria-label="Select agent">
            <option value="">All agents</option>
            {agents.map(a => <option key={a} value={a}>{a}</option>)}
          </select>
        }
      />
      {loading ? <SkeletonCard /> : error
        ? <ErrorState msg="Could not reach /training_log." onRetry={() => { setLoading(true); refresh(agent); }} />
        : !data.length ? <EmptyState icon="📈" msg="No training data yet. Start a training run." />
        : (
          <ResponsiveContainer width="100%" height={210}>
            <LineChart data={data} margin={{ top: 4, right: 16, left: -8, bottom: 0 }}>
              <defs>
                <linearGradient id="scoreGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={C.blue} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={C.blue} stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="ep" stroke={C.dim} tick={{ fill: C.muted, fontSize: 11 }} label={{ value: "Episode", position: "insideBottomRight", offset: -6, fill: C.muted, fontSize: 11 }} />
              <YAxis domain={[0, 1]} stroke={C.dim} tick={{ fill: C.muted, fontSize: 11 }} />
              <Tooltip {...TIP} formatter={(v, n) => [typeof v === "number" ? v.toFixed(4) : v, n]} />
              <Legend wrapperStyle={{ color: C.muted, fontSize: 11 }} />
              <Line type="monotone" dataKey="score" stroke={C.blue} strokeWidth={2.5}
                dot={{ r: 3.5, fill: C.blue, strokeWidth: 0 }} name="Grader Score" />
              {[
                { y: 0.6097, color: C.orange, label: "Llama-8B ZS" },
                { y: 0.3750, color: C.red,    label: "All-Allow"   },
              ].map(b => (
                <Line key={b.label} data={data.map(d => ({ ...d, [b.label]: b.y }))}
                  dataKey={b.label} stroke={b.color} strokeDasharray="5 4"
                  strokeWidth={1.5} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        )}
    </Card>
  );
}

function ActionDistPanel({ sessionId }) {
  const [distData, setDist] = useState([]);
  const [loading, setL]     = useState(false);
  const [error, setE]       = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    let dead = false;
    setL(true);
    fetchJSON(`${BASE_URL}/reward_breakdown?session_id=${sessionId}`).then(bd => {
      if (dead) return;
      if (!bd) { setE(true); setL(false); return; }
      const b = bd.breakdown || {};
      setE(false);
      setDist([
        { name: "Allow",         value: b.correct_allows?.count  ?? 0, color: C.green  },
        { name: "Correct Block", value: b.correct_blocks?.count  ?? 0, color: C.blue   },
        { name: "Missed Attack", value: b.missed_attacks?.count  ?? 0, color: C.red    },
        { name: "Over-Block",    value: b.over_blocks?.count     ?? 0, color: C.orange },
      ].filter(d => d.value > 0));
      setL(false);
    });
    return () => { dead = true; };
  }, [sessionId]);

  const total = distData.reduce((s, d) => s + d.value, 0);

  return (
    <Card accent={C.purple}>
      <CardHead title="Action Distribution" right={<Pill color={C.purple}>Last Episode</Pill>} />
      {!sessionId ? <EmptyState icon="🥧" msg="Paste a session ID to see action distribution." />
        : loading ? <SkeletonCard />
        : error   ? <ErrorState msg="Could not load reward breakdown." />
        : !distData.length ? <EmptyState icon="○" msg="No action data for this session." />
        : (
          <ResponsiveContainer width="100%" height={210}>
            <PieChart>
              <Pie data={distData} cx="50%" cy="50%" innerRadius={52} outerRadius={82}
                paddingAngle={4} dataKey="value"
                label={({ name, value }) => `${name} (${((value / total) * 100).toFixed(0)}%)`}
                labelLine={{ stroke: C.dim }}>
                {distData.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Pie>
              <Tooltip {...TIP} />
            </PieChart>
          </ResponsiveContainer>
        )}
    </Card>
  );
}

function TaskCompPanel({ agentName }) {
  const TASKS     = ["basic_threat_detection","context_aware_policy","multiturn_adversarial","adversarial_adaptation"];
  const SHORT     = ["Task 1","Task 2","Task 3","Task 4"];
  const build     = sc => TASKS.map((t, i) => ({ task: SHORT[i], score: sc[i], base: BASELINES["Llama-8B ZS"][i] }));

  const [data, setData]     = useState(build([null,null,null,null]));
  const [loading, setL]     = useState(false);
  const [error, setE]       = useState(false);

  useEffect(() => {
    if (!agentName) return;
    let dead = false;
    setL(true);
    fetchJSON(`${BASE_URL}/training_log?agent_name=${encodeURIComponent(agentName)}`).then(log => {
      if (dead) return;
      if (!log) { setE(true); setL(false); return; }
      setE(false);
      const entries = log.entries ?? [];
      setData(build(TASKS.map(t => {
        const te = entries.filter(e => e.task_id === t);
        return te.length ? te[te.length - 1].grader_score : null;
      })));
      setL(false);
    });
    return () => { dead = true; };
  }, [agentName]);

  return (
    <Card accent={C.green}>
      <CardHead
        title="Task Comparison"
        right={agentName ? <Pill color={C.green}>{agentName}</Pill> : null}
      />
      {!agentName ? <EmptyState icon="📊" msg="Enter an agent name to compare across tasks." />
        : loading ? <SkeletonCard />
        : error   ? <ErrorState msg="Could not load agent training log." />
        : (
          <ResponsiveContainer width="100%" height={210}>
            <BarChart data={data} margin={{ top: 4, right: 16, left: -8, bottom: 0 }} barGap={5}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="task" stroke={C.dim} tick={{ fill: C.muted, fontSize: 11 }} />
              <YAxis domain={[0, 1]} stroke={C.dim} tick={{ fill: C.muted, fontSize: 11 }} />
              <Tooltip {...TIP} formatter={v => v != null ? v.toFixed(4) : "—"} />
              <Legend wrapperStyle={{ color: C.muted, fontSize: 11 }} />
              <Bar dataKey="score" fill={C.blue}   name="Your Agent"  radius={[5,5,0,0]} />
              <Bar dataKey="base"  fill={C.orange}  name="Llama-8B ZS" radius={[5,5,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        )}
    </Card>
  );
}

function AdversaryPanel({ sessionId }) {
  const [traj, setTraj]     = useState([]);
  const [loading, setL]     = useState(false);
  const [error, setE]       = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    let dead = false;
    setL(true);
    fetchJSON(`${BASE_URL}/adversary_state?session_id=${sessionId}`).then(d => {
      if (dead) return;
      if (!d) { setE(true); setL(false); return; }
      if (d.adversary_trajectory) setTraj(d.adversary_trajectory);
      setL(false);
    });
    return () => { dead = true; };
  }, [sessionId]);

  const iColors = ["#22c55e","#86efac","#fbbf24","#f97316","#ef4444","#a78bfa"];
  const iLabels = ["Safe","Low","Med","Elev","High","Crit"];
  const topics    = new Set(traj.map(t => t.topic_idx)).size;
  const correct   = traj.filter(t => t.correct).length;
  const accuracy  = traj.length ? Math.round(correct / traj.length * 100) : 0;
  const accColor  = accuracy >= 70 ? C.green : accuracy >= 40 ? C.orange : C.red;

  return (
    <Card accent={C.orange}>
      <CardHead
        title="Task 4: Adversary Trajectory"
        right={topics > 0 ? <Pill color={C.purple}>{topics} topics</Pill> : null}
      />
      {!sessionId ? <EmptyState icon="🎯" msg="Run a Task 4 episode to see the adversary trajectory." />
        : loading ? <SkeletonCard />
        : error   ? <ErrorState msg="Could not load adversary state." />
        : !traj.length ? <EmptyState icon="○" msg="No trajectory data." />
        : (
          <>
            {/* Metric row */}
            <div style={{ display: "flex", gap: 28, paddingBottom: 14, marginBottom: 14, borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
              {[
                { label: "Turns",    val: traj.length, color: C.sub    },
                { label: "Topics",   val: topics,      color: C.purple  },
                { label: "Accuracy", val: `${accuracy}%`, color: accColor },
              ].map(s => (
                <div key={s.label}>
                  <div style={{ fontSize: 26, fontWeight: 700, color: s.color, fontVariantNumeric: "tabular-nums", lineHeight: 1.1 }}>{s.val}</div>
                  <div style={{ fontSize: 10, color: C.muted, textTransform: "uppercase", letterSpacing: "0.07em", marginTop: 3 }}>{s.label}</div>
                </div>
              ))}
            </div>
            {/* Table */}
            <div style={{ maxHeight: 200, overflowY: "auto" }}>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
                <thead>
                  <tr style={{ position: "sticky", top: 0, background: "rgba(7,11,16,0.95)", zIndex: 1 }}>
                    {["Turn","Topic","Intensity","Action","✓"].map(h => (
                      <th key={h} style={{ padding: "7px 10px", color: C.dim, textAlign: "left", borderBottom: "1px solid rgba(255,255,255,0.07)", fontWeight: 500, textTransform: "uppercase", fontSize: 9.5, letterSpacing: "0.07em" }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {traj.map((s, i) => (
                    <tr key={i} className="ga-row" style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                      <td style={{ padding: "7px 10px", color: C.muted, fontVariantNumeric: "tabular-nums" }}>{s.turn}</td>
                      <td style={{ padding: "7px 10px", color: C.sub }}>T{s.topic_idx}</td>
                      <td style={{ padding: "7px 10px" }}>
                        <span style={{ padding: "2px 8px", borderRadius: 20, background: iColors[s.intensity] + "20", color: iColors[s.intensity], fontSize: 10, fontWeight: 600, border: `1px solid ${iColors[s.intensity]}33` }}>
                          {iLabels[s.intensity]}
                        </span>
                      </td>
                      <td style={{ padding: "7px 10px" }}>
                        <span style={{ padding: "2px 8px", borderRadius: 20, fontSize: 10, fontWeight: 600,
                          background: s.agent_action === "allow" ? "#16a34a20" : "#dc262620",
                          color:      s.agent_action === "allow" ? C.green : C.red,
                          border:     `1px solid ${s.agent_action === "allow" ? "#16a34a44" : "#dc262644"}` }}>
                          {s.agent_action}
                        </span>
                      </td>
                      <td style={{ padding: "7px 10px", fontSize: 15, fontWeight: 700, color: s.correct ? C.green : C.red }}>{s.correct ? "✓" : "✗"}</td>
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

function LeaderboardPanel() {
  const [entries, setEntries] = useState([]);
  const [task, setTask]       = useState("basic_threat_detection");
  const [loading, setL]       = useState(true);
  const [error, setE]         = useState(false);

  async function load() {
    const d = await fetchJSON(`${BASE_URL}/leaderboard`);
    if (!d) { setE(true); setL(false); return; }
    setE(false);
    setEntries(((d.leaderboard ?? []).filter(e => e.task_id === task)).slice(0, 10));
    setL(false);
  }

  useEffect(() => {
    let dead = false;
    setL(true);
    load();
    const id = setInterval(() => !dead && load(), REFRESH_MS * 2);
    return () => { dead = true; clearInterval(id); };
  }, [task]);

  const maxScore = entries[0]?.score ?? 1;
  const medals = ["🥇", "🥈", "🥉"];

  return (
    <Card>
      <CardHead
        title="Leaderboard"
        right={
          <select className="ga-select" style={S.select} value={task}
            onChange={e => setTask(e.target.value)} aria-label="Select task">
            <option value="basic_threat_detection">Task 1 — Basic Threat</option>
            <option value="context_aware_policy">Task 2 — Context Policy</option>
            <option value="multiturn_adversarial">Task 3 — Multiturn Adv.</option>
            <option value="adversarial_adaptation">Task 4 — Adv. Adapt.</option>
          </select>
        }
      />
      {loading ? <SkeletonCard /> : error
        ? <ErrorState msg="Could not load leaderboard." onRetry={() => { setL(true); load(); }} />
        : !entries.length ? <EmptyState icon="🏆" msg="No submissions yet. Submit a score to appear here." />
        : (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr>
                {["Rank","Agent","Score"].map(h => (
                  <th key={h} style={{ padding: "7px 10px", color: C.dim, textAlign: "left", borderBottom: "1px solid rgba(255,255,255,0.07)", fontWeight: 500, fontSize: 10, textTransform: "uppercase", letterSpacing: "0.07em" }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {entries.map((e, i) => (
                <tr key={i} className="ga-row" style={{ borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                  <td style={{ padding: "9px 10px", fontWeight: 700, fontSize: 14 }}>
                    {medals[i] ?? <span style={{ color: C.muted }}>#{i + 1}</span>}
                  </td>
                  <td style={{ padding: "9px 10px", color: i === 0 ? C.text : C.sub, fontWeight: i === 0 ? 600 : 400 }}>
                    {e.agent_name || "—"}
                  </td>
                  <td style={{ padding: "9px 10px" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
                      <div style={{ flex: 1, maxWidth: 90, height: 5, background: "rgba(255,255,255,0.08)", borderRadius: 4, overflow: "hidden" }}>
                        <div className="ga-fill" style={{ height: "100%", borderRadius: 4, width: `${((e.score ?? 0) / maxScore) * 100}%`, background: i === 0 ? `linear-gradient(90deg, ${C.yellow}, ${C.orange})` : C.blue }} />
                      </div>
                      <span style={{ color: i === 0 ? C.yellow : C.sub, fontWeight: 700, fontVariantNumeric: "tabular-nums", fontSize: 12.5, minWidth: 50 }}>
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

// ── Task 4 Runner ─────────────────────────────────────────────────────────────

async function runTask4Episode() {
  const reset = await fetchJSON(`${BASE_URL}/reset?task_id=adversarial_adaptation`);
  if (!reset?.session_id) throw new Error("Reset failed — is the environment online?");
  const sid = reset.session_id;
  let obs = reset, done = false, steps = 0;
  while (!done && steps < 35) {
    const r = await fetch(`${BASE_URL}/step?session_id=${sid}`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt_id: obs.prompt_id, action_type: obs.user_risk_score > 0.5 ? "refuse" : "allow", reason: "demo", modified_prompt: null }),
      signal: AbortSignal.timeout(15000),
    });
    if (!r.ok) break;
    const d = await r.json();
    done = d.done; obs = d.observation || obs; steps++;
  }
  return { sessionId: sid, done, steps };
}

// ── Status Dot ────────────────────────────────────────────────────────────────

function StatusDot({ status }) {
  const col = status === "online" ? C.green : status === "offline" ? C.red : C.yellow;
  return (
    <div style={{ position: "relative", width: 10, height: 10, flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
      {status === "online" && <div className="ga-pulse" style={{ position: "absolute", inset: -4, borderRadius: "50%", background: C.green, opacity: 0.5 }} />}
      <div style={{ width: 8, height: 8, borderRadius: "50%", background: col, boxShadow: `0 0 8px ${col}88`, position: "relative" }} />
    </div>
  );
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

export default function Dashboard() {
  const [urlInput, setUrl]   = useState(BASE_URL);
  const [agent, setAgent]    = useState("guardrail_trl_agent");
  const [sid, setSid]        = useState("");
  const [t4sid, setT4sid]    = useState("");
  const [status, setStatus]  = useState("checking");
  const [ts, setTs]          = useState(new Date());
  const [running, setRunning]= useState(false);
  const [t4msg, setT4msg]    = useState("");

  function connect() { BASE_URL = urlInput.replace(/\/+$/, ""); setStatus("checking"); }

  async function runEpisode() {
    setRunning(true); setT4msg("Running episode…"); setT4sid("");
    try {
      const { sessionId, done, steps } = await runTask4Episode();
      setT4sid(sessionId);
      setT4msg(done ? `Done · ${steps} steps · ${sessionId.slice(0, 8)}…` : `Partial (${steps} steps)`);
    } catch (e) { setT4msg(`Error: ${e.message}`); }
    finally { setRunning(false); }
  }

  useEffect(() => {
    let dead = false;
    const check = async () => {
      const r = await fetchJSON(`${BASE_URL}/health`);
      if (!dead) { setStatus(r ? "online" : "offline"); setTs(new Date()); }
    };
    check();
    const id = setInterval(check, REFRESH_MS);
    return () => { dead = true; clearInterval(id); };
  }, []);

  const statusLabel = { online: "Online", offline: "Offline", checking: "Checking…" };

  return (
    <>
      <GlobalStyles />
      <div className="ga-root" style={S.root}>

        {/* ── Header ── */}
        <header style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20, paddingBottom: 18, borderBottom: "1px solid rgba(255,255,255,0.07)" }}>
          <div>
            <h1 className="ga-title-gradient" style={{ fontSize: 24, fontWeight: 800, letterSpacing: "-0.6px", margin: 0 }}>
              Sentinel
            </h1>
            <p style={{ margin: "5px 0 0", fontSize: 12.5, color: C.muted, letterSpacing: "0.01em" }}>
              Training Dashboard · Multi-Agent Interactions (Theme #1)
            </p>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <StatusDot status={status} />
            <span style={{ color: status === "offline" ? C.red : C.muted, fontSize: 12 }}>
              {statusLabel[status]}
            </span>
            <span style={{ color: C.dim, fontSize: 11 }}>· {ts.toLocaleTimeString()}</span>
          </div>
        </header>

        {/* ── URL Bar ── */}
        <div style={{ display: "flex", gap: 10, alignItems: "center", marginBottom: 16, padding: "9px 14px", background: "rgba(0,0,0,0.3)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 11 }}>
          <span style={{ color: C.dim, fontSize: 10, fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.08em", whiteSpace: "nowrap" }}>ENV</span>
          <input className="ga-input" style={{ ...S.inp, flex: 1 }} value={urlInput}
            onChange={e => setUrl(e.target.value)} onKeyDown={e => e.key === "Enter" && connect()}
            placeholder="https://varunventra-guardrail-arena.hf.space" aria-label="Environment URL" />
          <button className="ga-btn-blue" style={BTN} onClick={connect}>Connect</button>
        </div>

        {/* ── Stat Strip ── */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 10, marginBottom: 18 }}>
          {[
            { label: "Qwen-235B · Task 4",    val: "0.0000", color: C.red,    accent: C.red    },
            { label: "Q-learner · Task 4",     val: "0.9540", color: C.green,  accent: C.green  },
            { label: "Llama-8B ZS · Task 1",  val: "0.6097", color: C.muted,  accent: C.gray   },
            { label: "Llama-8B SFT · Task 1", val: "0.9504", color: C.blueL,  accent: C.blue   },
            { label: "Branching Convos",        val: "30",     color: C.sub,    accent: C.purple },
            { label: "Adversary States",        val: "180",    color: C.sub,    accent: C.orange },
          ].map(item => (
            <div key={item.label} className="ga-stat-tile" style={{
              padding: "12px 14px", borderRadius: 12,
              background: "rgba(255,255,255,0.045)",
              border: "1px solid rgba(255,255,255,0.08)",
              borderTop: `2px solid ${item.accent}`,
              boxShadow: `0 0 16px ${item.accent}10`,
            }}>
              <div style={{ fontSize: 9.5, color: C.dim, textTransform: "uppercase", letterSpacing: "0.08em", marginBottom: 6 }}>{item.label}</div>
              <div style={{ fontSize: 22, fontWeight: 800, color: item.color, fontVariantNumeric: "tabular-nums", lineHeight: 1 }}>{item.val}</div>
            </div>
          ))}
        </div>

        {/* ── Architecture Panel ── */}
        <Card style={{ marginBottom: 18 }}>
          <CardHead title="Multi-Agent Architecture" right={<Pill color={C.purple}>Theme #1: Multi-Agent Interactions</Pill>} />
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            {[
              {
                icon: "⚔",
                title: "Adversary: DeterministicAdversary FSM",
                titleColor: C.red,
                accent: C.red,
                lines: [
                  { text: "60 states (10 topics × 6 intensities) × 3 variants =", em: null },
                  { text: "180 observable states", em: C.text },
                  { text: "Adapts in real-time to defender actions", em: null },
                  { text: "allow → escalate  ·  block → back off", em: C.orange },
                ],
              },
              {
                icon: "🛡",
                title: "Defender: Trainable Safety Policy",
                titleColor: C.green,
                accent: C.green,
                lines: [
                  { text: "Observation: prompt + context + risk_score", em: null },
                  { text: "Actions: allow / refuse / modify / escalate", em: null },
                  { text: "+0.20 correct block  ·  −0.30 missed attack", em: C.blue },
                ],
              },
            ].map(box => (
              <div key={box.title} style={{ padding: "14px 16px", borderRadius: 12, background: "rgba(0,0,0,0.3)", border: "1px solid rgba(255,255,255,0.08)", borderLeft: `3px solid ${box.accent}` }}>
                <div style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 10 }}>
                  <span style={{ fontSize: 18 }}>{box.icon}</span>
                  <span style={{ color: box.titleColor, fontWeight: 700, fontSize: 12.5 }}>{box.title}</span>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                  {box.lines.map((l, i) => (
                    <span key={i} style={{ color: l.em ? l.em : C.muted, fontSize: 12, lineHeight: 1.6 }}>{l.text}</span>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </Card>

        {/* ── Controls ── */}
        <div style={{ display: "flex", gap: 14, marginBottom: 18, flexWrap: "wrap", alignItems: "flex-end" }}>
          {[
            { id: "agentIn", label: "Agent Name", val: agent, set: setAgent, ph: "e.g. guardrail_trl_agent" },
            { id: "sidIn",   label: "Session ID", sub: "for action breakdown", val: sid, set: setSid, ph: "Paste UUID from /reset" },
          ].map(f => (
            <div key={f.id} style={{ display: "flex", flexDirection: "column", gap: 6 }}>
              <label htmlFor={f.id} style={S.label}>
                {f.label}{f.sub && <span style={{ color: C.dim, fontWeight: 400, textTransform: "none", letterSpacing: 0 }}> — {f.sub}</span>}
              </label>
              <input id={f.id} className="ga-input" style={S.inp} value={f.val}
                onChange={e => f.set(e.target.value)} placeholder={f.ph} />
            </div>
          ))}

          {/* ── PRIMARY CTA ── */}
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            <span style={S.label}>Task 4 Demo</span>
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <button
                className="ga-cta"
                style={{ ...BTN, padding: "11px 24px", fontSize: 13.5, minWidth: 190, borderRadius: 10 }}
                onClick={runEpisode}
                disabled={running}
                aria-busy={running}
              >
                {running ? (
                  <><Spinner size={15} /> Running…</>
                ) : (
                  <><span style={{ fontSize: 15 }}>▶</span> Run Task 4 Episode</>
                )}
              </button>
              {t4msg && (
                <span style={{ fontSize: 11.5, color: t4msg.startsWith("Error") ? C.red : C.muted, maxWidth: 200, lineHeight: 1.5 }}>
                  {t4msg}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* ── 2×2 Chart Grid ── */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
          <RewardCurvePanel agentName={agent} />
          <ActionDistPanel  sessionId={sid} />
          <TaskCompPanel    agentName={agent} />
          <AdversaryPanel   sessionId={t4sid} />
        </div>

        {/* ── Leaderboard ── */}
        <LeaderboardPanel />

        {/* ── Footer ── */}
        <footer style={{ marginTop: 24, paddingTop: 16, borderTop: "1px solid rgba(255,255,255,0.07)", display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: 11, color: C.dim }}>
          <span>Sentinel · Meta × HuggingFace × Cerebral Valley OpenEnv Hackathon</span>
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
    minHeight: "100vh",
    fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
    color: C.text,
    padding: "28px 32px",
    maxWidth: 1280,
    margin: "0 auto",
    WebkitFontSmoothing: "antialiased",
    MozOsxFontSmoothing: "grayscale",
  },
  inp: {
    borderRadius: 8,
    padding: "9px 13px",
    fontSize: 13,
    minWidth: 240,
  },
  select: {
    borderRadius: 7,
    padding: "5px 9px",
    fontSize: 11,
  },
  label: {
    fontSize: 10.5,
    color: C.muted,
    textTransform: "uppercase",
    letterSpacing: "0.08em",
    fontWeight: 700,
  },
};
