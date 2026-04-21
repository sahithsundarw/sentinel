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

import React, { useState, useEffect, useRef } from "react";
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts";

// ── Config ────────────────────────────────────────────────────────────────────

let BASE_URL = "https://varunventra-guardrail-arena.hf.space";
const REFRESH_INTERVAL_MS = 5000;

const COLORS = {
  blue:   "#3b82f6",
  green:  "#22c55e",
  red:    "#ef4444",
  orange: "#f97316",
  purple: "#8b5cf6",
  yellow: "#eab308",
  gray:   "#6b7280",
  bg:     "#0a0a0a",
  card:   "#0f1117",
  border: "#1f2937",
  text:   "#e6edf3",
  muted:  "#9ca3af",
};

const TASK_LABELS = {
  basic_threat_detection: "Task 1: Basic Threat Detection",
  context_aware_policy:   "Task 2: Context-Aware Policy",
  multiturn_adversarial:  "Task 3: Multi-Turn Adversarial",
  adversarial_adaptation: "Task 4: Adversarial Adaptation",
};

// Real multi-task Colab results (Llama-3.1-8B zero-shot)
const BASELINES = {
  "All-Allow":    [0.3750, 0.4037, 0.1607, 0.1500],
  "All-Refuse":   [0.3534, 0.3460, 0.0688, 0.0000],
  "Llama-8B ZS":  [0.6097, 0.5493, 0.3988, 0.0000],
  "GPT-4o-mini":  [0.9216, 0.7512, 0.6120, 0.4820],
  "Qwen-235B":    [0.9857, 0.6862, 0.8275, 0.0000],
};

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

// ── Shared Sub-components ─────────────────────────────────────────────────────

function Spinner() {
  const [angle, setAngle] = useState(0);
  useEffect(() => {
    const id = setInterval(() => setAngle(a => (a + 30) % 360), 80);
    return () => clearInterval(id);
  }, []);
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "center", padding: "30px 0", gap: 10 }}>
      <svg width="20" height="20" viewBox="0 0 20 20" style={{ transform: `rotate(${angle}deg)` }}>
        <circle cx="10" cy="10" r="8" fill="none" stroke={COLORS.border} strokeWidth="2" />
        <path d="M10 2 A8 8 0 0 1 18 10" fill="none" stroke={COLORS.blue} strokeWidth="2" strokeLinecap="round" />
      </svg>
      <span style={{ color: COLORS.muted, fontSize: 12 }}>Loading…</span>
    </div>
  );
}

function EmptyState({ message }) {
  return (
    <div style={styles.empty}>
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" style={{ marginBottom: 8, opacity: 0.4 }}>
        <circle cx="12" cy="12" r="10" stroke={COLORS.muted} strokeWidth="1.5" />
        <path d="M12 8v4M12 16h.01" stroke={COLORS.muted} strokeWidth="1.5" strokeLinecap="round" />
      </svg>
      <div>{message}</div>
    </div>
  );
}

function ErrorState({ message }) {
  return (
    <div style={{ ...styles.empty, color: COLORS.red }}>
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" style={{ marginBottom: 8 }}>
        <circle cx="12" cy="12" r="10" stroke={COLORS.red} strokeWidth="1.5" />
        <path d="M15 9l-6 6M9 9l6 6" stroke={COLORS.red} strokeWidth="1.5" strokeLinecap="round" />
      </svg>
      <div>{message}</div>
    </div>
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

  useEffect(() => {
    let cancelled = false;
    async function refresh() {
      const all = await fetchJSON(`${BASE_URL}/training_log`);
      if (cancelled) return;
      if (!all) { setError(true); setLoading(false); return; }
      if (all?.agents) setAllAgents(all.agents);

      const target = selected || all?.agents?.[0] || "";
      if (!target) { setLoading(false); return; }

      const log = await fetchJSON(`${BASE_URL}/training_log?agent_name=${encodeURIComponent(target)}`);
      if (cancelled) return;
      setError(false);
      if (log?.entries) {
        setData(log.entries.map(e => ({
          episode: e.episode,
          score:   e.grader_score,
          reward:  e.cumulative_reward,
        })));
      }
      setLoading(false);
    }
    setLoading(true);
    refresh();
    const id = setInterval(refresh, REFRESH_INTERVAL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, [selected]);

  // Sync external agentName prop → local selection
  useEffect(() => { if (agentName) setSelected(agentName); }, [agentName]);

  function renderBody() {
    if (loading)      return <Spinner />;
    if (error)        return <ErrorState message="Could not reach /training_log endpoint." />;
    if (!data.length) return <EmptyState message="No training data yet. Start a training run." />;
    return (
      <ResponsiveContainer width="100%" height={220}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
          <XAxis dataKey="episode" stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} label={{ value: "Episode", position: "insideBottomRight", offset: -4, fill: COLORS.muted, fontSize: 11 }} />
          <YAxis domain={[0, 1]} stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} />
          <Tooltip
            contentStyle={{ backgroundColor: COLORS.card, border: `1px solid ${COLORS.border}`, color: COLORS.text }}
            formatter={(v, name) => [typeof v === "number" ? v.toFixed(4) : v, name]}
          />
          <Legend wrapperStyle={{ color: COLORS.muted, fontSize: 11 }} />
          <Line type="monotone" dataKey="score" stroke={COLORS.blue} strokeWidth={2} dot={{ r: 3 }} name="Grader Score" />
          {[
            { y: 0.3750, color: COLORS.red,    label: "All-Allow" },
            { y: 0.6097, color: COLORS.orange,  label: "Llama-8B ZS" },
          ].map(b => (
            <Line
              key={b.label}
              data={data.map(d => ({ ...d, [b.label]: b.y }))}
              dataKey={b.label}
              stroke={b.color}
              strokeDasharray="4 4"
              strokeWidth={1.5}
              dot={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    );
  }

  return (
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <span style={styles.cardTitle}>Live Reward Curve</span>
        <select
          style={styles.select}
          value={selected}
          onChange={e => setSelected(e.target.value)}
          aria-label="Select agent"
        >
          <option value="">All agents</option>
          {allAgents.map(a => <option key={a} value={a}>{a}</option>)}
        </select>
      </div>
      {renderBody()}
    </div>
  );
}

// 2. Action Distribution Pie Chart
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
        { name: "Allow",        value: b.correct_allows?.count  ?? 0, color: COLORS.green  },
        { name: "Correct Block",value: b.correct_blocks?.count  ?? 0, color: COLORS.blue   },
        { name: "Missed Attack",value: b.missed_attacks?.count  ?? 0, color: COLORS.red    },
        { name: "Over-Block",   value: b.over_blocks?.count     ?? 0, color: COLORS.orange },
      ].filter(d => d.value > 0));
      setLoading(false);
    }
    load();
    return () => { cancelled = true; };
  }, [sessionId]);

  const total = distData.reduce((s, d) => s + d.value, 0);

  function renderBody() {
    if (!sessionId)   return <EmptyState message="Complete an episode and paste its session ID above." />;
    if (loading)      return <Spinner />;
    if (error)        return <ErrorState message="Could not load reward breakdown." />;
    if (!distData.length) return <EmptyState message="No action data for this session." />;
    return (
      <ResponsiveContainer width="100%" height={220}>
        <PieChart>
          <Pie
            data={distData}
            cx="50%" cy="50%"
            innerRadius={55} outerRadius={85}
            paddingAngle={3}
            dataKey="value"
            label={({ name, value }) => `${name} (${((value / total) * 100).toFixed(0)}%)`}
            labelLine={{ stroke: COLORS.muted }}
          >
            {distData.map((entry, i) => <Cell key={i} fill={entry.color} />)}
          </Pie>
          <Tooltip contentStyle={{ backgroundColor: COLORS.card, border: `1px solid ${COLORS.border}`, color: COLORS.text }} />
        </PieChart>
      </ResponsiveContainer>
    );
  }

  return (
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <span style={styles.cardTitle}>Action Distribution</span>
        <span style={{ ...styles.badge, background: "#1e3a5f" }}>Last Episode</span>
      </div>
      {renderBody()}
    </div>
  );
}

// 3. Task Comparison Bar Chart — single fetch, not N fetches
function TaskComparisonPanel({ agentName }) {
  const tasks     = ["basic_threat_detection", "context_aware_policy", "multiturn_adversarial", "adversarial_adaptation"];
  const taskShort = ["Task 1", "Task 2", "Task 3", "Task 4"];

  function buildData(scores) {
    return tasks.map((t, i) => ({
      task:     taskShort[i],
      score:    scores[i],
      allAllow: BASELINES["All-Allow"][i],
      llama8b:  BASELINES["Llama-8B ZS"][i],
    }));
  }

  const [data, setData]     = useState(buildData([null, null, null, null]));
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState(false);

  useEffect(() => {
    if (!agentName) return;
    let cancelled = false;
    async function load() {
      setLoading(true);
      // Single fetch — server returns all task entries for the agent
      const log = await fetchJSON(`${BASE_URL}/training_log?agent_name=${encodeURIComponent(agentName)}`);
      if (cancelled) return;
      if (!log) { setError(true); setLoading(false); return; }
      setError(false);
      const entries = log.entries ?? [];
      const scores = tasks.map(t => {
        const taskEntries = entries.filter(e => e.task_id === t);
        return taskEntries.length ? taskEntries[taskEntries.length - 1].grader_score : null;
      });
      setData(buildData(scores));
      setLoading(false);
    }
    load();
  }, [agentName]);

  function renderBody() {
    if (!agentName) return <EmptyState message="Enter an agent name above to compare tasks." />;
    if (loading)    return <Spinner />;
    if (error)      return <ErrorState message="Could not load agent training log." />;
    return (
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
          <XAxis dataKey="task" stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} />
          <YAxis domain={[0, 1]} stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} />
          <Tooltip
            contentStyle={{ backgroundColor: COLORS.card, border: `1px solid ${COLORS.border}`, color: COLORS.text }}
            formatter={v => v != null ? v.toFixed(4) : "—"}
          />
          <Legend wrapperStyle={{ color: COLORS.muted, fontSize: 11 }} />
          <Bar dataKey="score"    fill={COLORS.blue}   name="Your Agent"  radius={[3, 3, 0, 0]} />
          <Bar dataKey="allAllow" fill={COLORS.gray}   name="All-Allow"   radius={[3, 3, 0, 0]} />
          <Bar dataKey="llama8b"  fill={COLORS.orange} name="Llama-8B ZS" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    );
  }

  return (
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <span style={styles.cardTitle}>Task Comparison</span>
        {agentName && <span style={{ ...styles.badge, background: "#1a2e1a" }}>{agentName}</span>}
      </div>
      {renderBody()}
    </div>
  );
}

// 4. Task 4 Adversary Trajectory Table
function AdversaryStatePanel({ sessionId }) {
  const [trajectory, setTrajectory] = useState([]);
  const [loading, setLoading]       = useState(false);
  const [error, setError]           = useState(false);

  useEffect(() => {
    if (!sessionId) return;
    let cancelled = false;
    async function load() {
      setLoading(true);
      const data = await fetchJSON(`${BASE_URL}/adversary_state?session_id=${sessionId}`);
      if (cancelled) return;
      if (!data) { setError(true); setLoading(false); return; }
      if (data?.adversary_trajectory) setTrajectory(data.adversary_trajectory);
      setLoading(false);
    }
    load();
    return () => { cancelled = true; };
  }, [sessionId]);

  const intensityColors = ["#22c55e", "#86efac", "#fbbf24", "#f97316", "#ef4444", "#7c3aed"];
  const intensityLabels = ["Safe (0)", "Low (1)", "Med (2)", "Elev (3)", "High (4)", "Crit (5)"];

  const topicsHit    = new Set(trajectory.map(t => t.topic_idx)).size;
  const correctCount = trajectory.filter(t => t.correct).length;
  const accuracy     = trajectory.length ? ((correctCount / trajectory.length) * 100).toFixed(0) : 0;

  function renderBody() {
    if (loading)          return <Spinner />;
    if (error)            return <ErrorState message="Could not load adversary state." />;
    if (!trajectory.length) return <EmptyState message="Complete a Task 4 episode to see adversary trajectory." />;
    return (
      <>
        {/* Summary row */}
        <div style={{ display: "flex", gap: 20, marginBottom: 12 }}>
          {[
            { label: "Turns",    value: trajectory.length,   color: COLORS.text  },
            { label: "Topics",   value: topicsHit,           color: COLORS.purple },
            { label: "Accuracy", value: `${accuracy}%`,      color: correctCount / trajectory.length >= 0.7 ? COLORS.green : COLORS.red },
          ].map(s => (
            <div key={s.label} style={{ textAlign: "center" }}>
              <div style={{ fontSize: 18, fontWeight: 700, color: s.color }}>{s.value}</div>
              <div style={{ fontSize: 10, color: COLORS.muted, textTransform: "uppercase", letterSpacing: "0.05em" }}>{s.label}</div>
            </div>
          ))}
        </div>
        <div style={{ overflowX: "auto", maxHeight: 260, overflowY: "auto" }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
            <thead>
              <tr>
                {["Turn", "Topic", "Intensity", "Action", "✓"].map(h => (
                  <th key={h} style={{ padding: "6px 8px", color: COLORS.muted, textAlign: "left", borderBottom: `1px solid ${COLORS.border}`, position: "sticky", top: 0, background: COLORS.card }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {trajectory.map((step, i) => (
                <tr key={i} style={{ borderBottom: `1px solid ${COLORS.border}`, background: i % 2 === 0 ? "transparent" : "#0d1117" }}>
                  <td style={{ padding: "5px 8px", color: COLORS.muted }}>{step.turn}</td>
                  <td style={{ padding: "5px 8px", color: COLORS.text }}>T{step.topic_idx}</td>
                  <td style={{ padding: "5px 8px" }}>
                    <span style={{ display: "inline-block", padding: "2px 6px", borderRadius: 4, background: intensityColors[step.intensity] + "22", color: intensityColors[step.intensity], fontSize: 10, fontWeight: "bold" }}>
                      {intensityLabels[step.intensity]}
                    </span>
                  </td>
                  <td style={{ padding: "5px 8px" }}>
                    <span style={{ display: "inline-block", padding: "2px 6px", borderRadius: 4, background: step.agent_action === "allow" ? "#1a2e1a" : "#2d1b1b", color: step.agent_action === "allow" ? COLORS.green : COLORS.red, fontSize: 10 }}>
                      {step.agent_action}
                    </span>
                  </td>
                  <td style={{ padding: "5px 8px", color: step.correct ? COLORS.green : COLORS.red, fontWeight: "bold" }}>
                    {step.correct ? "✓" : "✗"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </>
    );
  }

  return (
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <span style={styles.cardTitle}>Task 4: Adversary Trajectory</span>
        {topicsHit > 0 && (
          <span style={{ ...styles.badge, background: "#2d1b69", color: COLORS.purple }}>
            {topicsHit} topics
          </span>
        )}
      </div>
      {renderBody()}
    </div>
  );
}

// 5. Agent Leaderboard
function LeaderboardPanel() {
  const [entries, setEntries] = useState([]);
  const [task, setTask]       = useState("basic_threat_detection");
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(false);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      const data = await fetchJSON(`${BASE_URL}/leaderboard`);
      if (cancelled) return;
      if (!data) { setError(true); setLoading(false); return; }
      setError(false);
      const filtered = (data.leaderboard ?? []).filter(e => e.task_id === task);
      setEntries(filtered.slice(0, 10));
      setLoading(false);
    }
    load();
    const id = setInterval(load, REFRESH_INTERVAL_MS * 2);
    return () => { cancelled = true; clearInterval(id); };
  }, [task]);

  const maxScore = entries[0]?.score ?? 1;

  function renderBody() {
    if (loading)         return <Spinner />;
    if (error)           return <ErrorState message="Could not load leaderboard." />;
    if (!entries.length) return <EmptyState message="No submissions yet for this task." />;
    return (
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
        <thead>
          <tr>
            {["Rank", "Agent", "Score"].map(h => (
              <th key={h} style={{ padding: "6px 8px", color: COLORS.muted, textAlign: "left", borderBottom: `1px solid ${COLORS.border}` }}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {entries.map((e, i) => (
            <tr key={i} style={{ borderBottom: `1px solid ${COLORS.border}` }}>
              <td style={{ padding: "5px 8px", color: i === 0 ? COLORS.yellow : COLORS.muted, fontWeight: "bold" }}>#{i + 1}</td>
              <td style={{ padding: "5px 8px", color: COLORS.text }}>{e.agent_name || "—"}</td>
              <td style={{ padding: "5px 8px" }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <div style={{ width: `${((e.score ?? 0) / maxScore) * 80}px`, height: 6, background: i === 0 ? COLORS.yellow : COLORS.blue, borderRadius: 4, minWidth: 4, transition: "width 0.3s ease" }} />
                  <span style={{ color: i === 0 ? COLORS.yellow : COLORS.text, fontWeight: "bold" }}>
                    {(e.score ?? 0).toFixed(4)}
                  </span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }

  return (
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <span style={styles.cardTitle}>Leaderboard</span>
        <select style={styles.select} value={task} onChange={e => setTask(e.target.value)} aria-label="Select task">
          <option value="basic_threat_detection">Task 1</option>
          <option value="context_aware_policy">Task 2</option>
          <option value="multiturn_adversarial">Task 3</option>
          <option value="adversarial_adaptation">Task 4</option>
        </select>
      </div>
      {renderBody()}
    </div>
  );
}

// ── Task 4 Episode Runner ─────────────────────────────────────────────────────

async function runTask4Episode() {
  const resetRes = await fetchJSON(`${BASE_URL}/reset?task_id=adversarial_adaptation`);
  if (!resetRes?.session_id) throw new Error("Reset failed — is the environment online?");
  const sid = resetRes.session_id;
  let obs  = resetRes;
  let done = false;
  let steps = 0;
  while (!done && steps < 35) {           // 30-turn episodes + safety margin
    const action = {
      prompt_id:       obs.prompt_id,
      action_type:     obs.user_risk_score > 0.5 ? "refuse" : "allow",
      reason:          "dashboard demo",
      modified_prompt: null,
    };
    const res = await fetch(`${BASE_URL}/step?session_id=${sid}`, {
      method:  "POST",
      headers: { "Content-Type": "application/json" },
      body:    JSON.stringify(action),
      signal:  AbortSignal.timeout(15000),
    });
    if (!res.ok) break;
    const data = await res.json();
    done = data.done;
    obs  = data.observation || obs;
    steps++;
  }
  return { sessionId: sid, done, steps };
}

// ── Pulsing Status Dot ────────────────────────────────────────────────────────

function StatusDot({ status }) {
  const [pulse, setPulse] = useState(false);
  useEffect(() => {
    if (status !== "online") return;
    const id = setInterval(() => setPulse(p => !p), 1500);
    return () => clearInterval(id);
  }, [status]);

  const color = status === "online" ? COLORS.green : status === "offline" ? COLORS.red : COLORS.yellow;
  return (
    <div style={{ position: "relative", width: 10, height: 10, flexShrink: 0 }}>
      {status === "online" && (
        <div style={{
          position: "absolute", inset: -3, borderRadius: "50%",
          background: COLORS.green,
          opacity: pulse ? 0 : 0.25,
          transition: "opacity 1.4s ease",
        }} />
      )}
      <div style={{ width: 10, height: 10, borderRadius: "50%", background: color, position: "relative" }} />
    </div>
  );
}

// ── Main Dashboard ────────────────────────────────────────────────────────────

export default function TrainingDashboard() {
  const [urlInput, setUrlInput]         = useState(BASE_URL);
  const [agentName, setAgentName]       = useState("guardrail_trl_agent");
  const [sessionId, setSessionId]       = useState("");
  const [task4SessionId, setTask4SessionId] = useState("");
  const [envStatus, setEnvStatus]       = useState("checking");
  const [lastRefresh, setLastRefresh]   = useState(new Date());
  const [task4Running, setTask4Running] = useState(false);
  const [task4Msg, setTask4Msg]         = useState("");

  function handleConnect() {
    BASE_URL = urlInput.replace(/\/+$/, "");
    setEnvStatus("checking");
  }

  async function handleRunTask4() {
    setTask4Running(true);
    setTask4Msg("Running Task 4 episode…");
    setTask4SessionId("");
    try {
      const { sessionId: sid, done, steps } = await runTask4Episode();
      setTask4SessionId(sid);
      setTask4Msg(
        done
          ? `Done in ${steps} steps · Session: ${sid.slice(0, 8)}…`
          : `Partial (${steps} steps) · Session: ${sid.slice(0, 8)}…`
      );
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
      if (!cancelled) {
        setEnvStatus(r ? "online" : "offline");
        setLastRefresh(new Date());
      }
    }
    checkHealth();
    const id = setInterval(checkHealth, REFRESH_INTERVAL_MS);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  return (
    <div style={styles.root}>

      {/* ── Header ── */}
      <div style={styles.header}>
        <div>
          <h1 style={styles.title}>Guardrail Arena</h1>
          <p style={styles.subtitle}>Training Dashboard — Multi-Agent Interactions (Theme #1)</p>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <StatusDot status={envStatus} />
          <span style={{ color: COLORS.muted, fontSize: 12 }}>
            {envStatus === "online" ? "Online" : envStatus === "offline" ? "Offline" : "Checking…"}
          </span>
          <span style={{ color: COLORS.border, fontSize: 12 }}>·</span>
          <span style={{ color: COLORS.muted, fontSize: 11 }}>{lastRefresh.toLocaleTimeString()}</span>
        </div>
      </div>

      {/* ── URL Bar ── */}
      <div style={styles.urlBar}>
        <span style={{ color: COLORS.muted, fontSize: 11, whiteSpace: "nowrap" }}>Environment URL</span>
        <input
          style={{ ...styles.input, flex: 1 }}
          value={urlInput}
          onChange={e => setUrlInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleConnect()}
          placeholder="https://varunventra-guardrail-arena.hf.space"
          aria-label="Environment URL"
        />
        <button style={styles.btnPrimary} onClick={handleConnect}>Connect</button>
      </div>

      {/* ── Key Numbers Strip ── */}
      <div style={styles.infoBar}>
        {[
          { label: "Qwen-235B (Task 4)",   value: "0.0000", color: COLORS.red    },
          { label: "Q-learner (Task 4)",    value: "0.9540", color: COLORS.green  },
          { label: "Llama-8B ZS (Task 1)", value: "0.6097", color: COLORS.muted  },
          { label: "Llama-8B SFT (Task 1)",value: "0.9504", color: COLORS.blue   },
          { label: "Branching Convos",      value: "30",     color: COLORS.text   },
          { label: "Adversary States",      value: "180",    color: COLORS.text   },
        ].map(item => (
          <div key={item.label} style={styles.infoItem}>
            <span style={styles.infoLabel}>{item.label}</span>
            <span style={{ ...styles.infoValue, color: item.color }}>{item.value}</span>
          </div>
        ))}
      </div>

      {/* ── Multi-Agent Architecture Panel ── */}
      <div style={{ ...styles.card, marginBottom: 16 }}>
        <div style={styles.cardHeader}>
          <span style={styles.cardTitle}>Multi-Agent Architecture</span>
          <span style={{ ...styles.badge, background: "#1a1a2e", color: COLORS.purple }}>Theme #1: Multi-Agent Interactions</span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
          <div style={styles.agentBox}>
            <div style={{ color: COLORS.red, fontWeight: 700, marginBottom: 8, fontSize: 13 }}>Adversary: DeterministicAdversary FSM</div>
            <div style={{ color: COLORS.muted, fontSize: 12, lineHeight: 1.7 }}>
              60 states (10 topics × 6 intensities) × 3 surface variants = 180 observable states<br />
              Adapts in real-time to defender actions<br />
              <span style={{ color: COLORS.orange }}>allow → escalate · block → back off</span>
            </div>
          </div>
          <div style={styles.agentBox}>
            <div style={{ color: COLORS.green, fontWeight: 700, marginBottom: 8, fontSize: 13 }}>Defender: Trainable Safety Policy</div>
            <div style={{ color: COLORS.muted, fontSize: 12, lineHeight: 1.7 }}>
              Observation: prompt + context + risk_score<br />
              Actions: allow / refuse / modify / escalate<br />
              <span style={{ color: COLORS.blue }}>+0.20 correct block · −0.30 missed attack</span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Controls ── */}
      <div style={styles.controls}>
        <div style={styles.inputGroup}>
          <label htmlFor="agentNameInput" style={styles.label}>Agent Name</label>
          <input
            id="agentNameInput"
            style={styles.input}
            value={agentName}
            onChange={e => setAgentName(e.target.value)}
            placeholder="e.g. guardrail_trl_agent"
          />
        </div>
        <div style={styles.inputGroup}>
          <label htmlFor="sessionIdInput" style={styles.label}>Session ID <span style={{ color: COLORS.border }}>(for action breakdown)</span></label>
          <input
            id="sessionIdInput"
            style={styles.input}
            value={sessionId}
            onChange={e => setSessionId(e.target.value)}
            placeholder="Paste UUID from /reset"
          />
        </div>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Task 4 Demo</label>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button
              style={{ ...styles.btnPrimary, background: task4Running ? COLORS.gray : COLORS.purple, cursor: task4Running ? "not-allowed" : "pointer" }}
              onClick={handleRunTask4}
              disabled={task4Running}
              aria-busy={task4Running}
            >
              {task4Running ? "Running…" : "▶ Run Task 4 Episode"}
            </button>
            {task4Msg && (
              <span style={{ fontSize: 11, color: task4Msg.startsWith("Error") ? COLORS.red : COLORS.muted }}>
                {task4Msg}
              </span>
            )}
          </div>
        </div>
      </div>

      {/* ── Main Grid ── */}
      <div style={styles.grid}>
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
      <div style={styles.footer}>
        <span>Guardrail Arena · Meta × HuggingFace × Cerebral Valley OpenEnv Hackathon</span>
        <div style={{ display: "flex", gap: 16 }}>
          <a href={`${BASE_URL}/docs`} target="_blank" rel="noopener noreferrer" style={{ color: COLORS.blue }}>API Docs</a>
          <a href="https://github.com/sahithsundarw/sentinel" target="_blank" rel="noopener noreferrer" style={{ color: COLORS.muted }}>GitHub</a>
        </div>
      </div>
    </div>
  );
}

// ── Styles ────────────────────────────────────────────────────────────────────

const styles = {
  root: {
    backgroundColor: COLORS.bg,
    minHeight: "100vh",
    fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
    color: COLORS.text,
    padding: "20px 24px",
    maxWidth: 1200,
    margin: "0 auto",
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 14,
    paddingBottom: 14,
    borderBottom: `1px solid ${COLORS.border}`,
  },
  title: {
    margin: 0,
    fontSize: 22,
    fontWeight: 700,
    color: COLORS.text,
    letterSpacing: "-0.3px",
  },
  subtitle: {
    margin: "4px 0 0",
    fontSize: 13,
    color: COLORS.muted,
  },
  urlBar: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    padding: "8px 12px",
    background: COLORS.card,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 8,
    marginBottom: 12,
  },
  infoBar: {
    display: "flex",
    gap: 0,
    background: COLORS.card,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 8,
    marginBottom: 16,
    overflow: "hidden",
  },
  infoItem: {
    display: "flex",
    flexDirection: "column",
    gap: 3,
    padding: "10px 20px",
    borderRight: `1px solid ${COLORS.border}`,
    flex: 1,
  },
  infoLabel: {
    fontSize: 10,
    color: COLORS.muted,
    textTransform: "uppercase",
    letterSpacing: "0.05em",
    whiteSpace: "nowrap",
  },
  infoValue: {
    fontSize: 18,
    fontWeight: 700,
    color: COLORS.text,
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
    gap: 5,
  },
  label: {
    fontSize: 11,
    color: COLORS.muted,
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  input: {
    background: COLORS.card,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 6,
    color: COLORS.text,
    padding: "7px 10px",
    fontSize: 13,
    outline: "none",
    minWidth: 240,
    transition: "border-color 0.15s ease",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(2, 1fr)",
    gap: 16,
  },
  card: {
    background: COLORS.card,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 8,
    padding: 16,
  },
  agentBox: {
    background: "#0d1117",
    borderRadius: 6,
    padding: "12px 16px",
    border: `1px solid ${COLORS.border}`,
  },
  cardHeader: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  cardTitle: {
    fontSize: 14,
    fontWeight: 600,
    color: COLORS.text,
  },
  badge: {
    fontSize: 10,
    padding: "2px 8px",
    borderRadius: 12,
    color: COLORS.text,
  },
  select: {
    background: COLORS.card,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 4,
    color: COLORS.text,
    padding: "3px 6px",
    fontSize: 11,
    outline: "none",
    cursor: "pointer",
  },
  empty: {
    color: COLORS.muted,
    fontSize: 12,
    textAlign: "center",
    padding: "30px 0",
    fontStyle: "italic",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    gap: 4,
  },
  footer: {
    marginTop: 20,
    paddingTop: 14,
    borderTop: `1px solid ${COLORS.border}`,
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    fontSize: 11,
    color: COLORS.muted,
  },
  btnPrimary: {
    background: COLORS.blue,
    border: "none",
    borderRadius: 6,
    color: "#fff",
    padding: "7px 14px",
    fontSize: 12,
    fontWeight: 600,
    cursor: "pointer",
    whiteSpace: "nowrap",
  },
};
