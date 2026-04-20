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

import React, { useState, useEffect, useCallback } from "react";
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ScatterChart, Scatter, ZAxis,
} from "recharts";

// ── Config ────────────────────────────────────────────────────────────────────

// Module-level mutable base URL — updated when user changes the URL input
let BASE_URL = "https://varunventra-guardrail-arena.hf.space";
const REFRESH_INTERVAL_MS = 5000;

const COLORS = {
  blue: "#3b82f6",
  green: "#22c55e",
  red: "#ef4444",
  orange: "#f97316",
  purple: "#8b5cf6",
  yellow: "#eab308",
  gray: "#6b7280",
  bg: "#0a0a0a",
  card: "#0f1117",
  border: "#1f2937",
  text: "#e6edf3",
  muted: "#9ca3af",
};

const TASK_LABELS = {
  basic_threat_detection: "Task 1: Basic Threat Detection",
  context_aware_policy: "Task 2: Context-Aware Policy",
  multiturn_adversarial: "Task 3: Multi-Turn Adversarial",
  adversarial_adaptation: "Task 4: Adversarial Adaptation",
};

const BASELINES = {
  "All-Allow":   [0.3750, 0.4037, 0.1607, 0.1500],
  "All-Refuse":  [0.3534, 0.3460, 0.0688, 0.0000],
  "Llama-8B ZS": [0.5428, 0.5143, 0.4746, 0.0000],
  "GPT-4o-mini": [0.9216, 0.7512, 0.6120, 0.4820],
  "Qwen-235B":   [0.9857, 0.6862, 0.8275, 0.0000],
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

// ── Components ────────────────────────────────────────────────────────────────

// 1. Live Reward Curve — polls /training_log every 5s
function RewardCurvePanel({ agentName }) {
  const [data, setData] = useState([]);
  const [allAgents, setAllAgents] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(agentName || "");

  useEffect(() => {
    async function refresh() {
      const all = await fetchJSON(`${BASE_URL}/training_log`);
      if (all?.agents) setAllAgents(all.agents);

      const target = selectedAgent || (all?.agents?.[0] ?? "");
      if (!target) return;
      const log = await fetchJSON(`${BASE_URL}/training_log?agent_name=${encodeURIComponent(target)}`);
      if (log?.entries) {
        setData(log.entries.map(e => ({
          episode: e.episode,
          score: e.grader_score,
          reward: e.cumulative_reward,
        })));
      }
    }
    refresh();
    const interval = setInterval(refresh, REFRESH_INTERVAL_MS);
    return () => clearInterval(interval);
  }, [selectedAgent]);

  return (
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <span style={styles.cardTitle}>Live Reward Curve</span>
        <select
          style={styles.select}
          value={selectedAgent}
          onChange={e => setSelectedAgent(e.target.value)}
        >
          <option value="">All agents</option>
          {allAgents.map(a => <option key={a} value={a}>{a}</option>)}
        </select>
      </div>
      {data.length === 0 ? (
        <div style={styles.empty}>No training data yet. Start a training run.</div>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
            <XAxis dataKey="episode" stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} />
            <YAxis domain={[0, 1]} stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} />
            <Tooltip
              contentStyle={{ backgroundColor: COLORS.card, border: `1px solid ${COLORS.border}`, color: COLORS.text }}
              formatter={(v, name) => [typeof v === "number" ? v.toFixed(4) : v, name]}
            />
            <Legend wrapperStyle={{ color: COLORS.muted, fontSize: 11 }} />
            <Line type="monotone" dataKey="score" stroke={COLORS.blue} strokeWidth={2} dot={{ r: 3 }} name="Grader Score" />
            {/* Baselines */}
            {[
              { y: 0.3750, color: COLORS.red, label: "All-Allow" },
              { y: 0.5428, color: COLORS.orange, label: "Llama-8B" },
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
      )}
    </div>
  );
}

// 2. Action Distribution Pie Chart
function ActionDistributionPanel({ sessionId }) {
  const [distData, setDistData] = useState([]);

  useEffect(() => {
    async function load() {
      if (!sessionId) return;
      const bd = await fetchJSON(`${BASE_URL}/reward_breakdown?session_id=${sessionId}`);
      if (!bd?.breakdown) return;
      const b = bd.breakdown;
      setDistData([
        { name: "Allow", value: b.correct_allows?.count ?? 0, color: COLORS.green },
        { name: "Refuse (correct)", value: b.correct_blocks?.count ?? 0, color: COLORS.blue },
        { name: "Missed attack", value: b.missed_attacks?.count ?? 0, color: COLORS.red },
        { name: "Over-block", value: b.over_blocks?.count ?? 0, color: COLORS.orange },
      ].filter(d => d.value > 0));
    }
    load();
  }, [sessionId]);

  const total = distData.reduce((s, d) => s + d.value, 0);

  return (
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <span style={styles.cardTitle}>Action Distribution</span>
        <span style={{ ...styles.badge, background: "#1e3a5f" }}>Last Episode</span>
      </div>
      {distData.length === 0 ? (
        <div style={styles.empty}>Complete an episode to see distribution.</div>
      ) : (
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
              {distData.map((entry, i) => (
                <Cell key={i} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip
              contentStyle={{ backgroundColor: COLORS.card, border: `1px solid ${COLORS.border}`, color: COLORS.text }}
            />
          </PieChart>
        </ResponsiveContainer>
      )}
    </div>
  );
}

// 3. Task Comparison Bar Chart
function TaskComparisonPanel({ agentName }) {
  const tasks = ["basic_threat_detection", "context_aware_policy", "multiturn_adversarial", "adversarial_adaptation"];
  const taskShort = ["Task 1", "Task 2", "Task 3", "Task 4"];

  const buildData = (scores) => tasks.map((t, i) => ({
    task: taskShort[i],
    score: scores[i],
    allAllow: BASELINES["All-Allow"][i],
    llama8b: BASELINES["Llama-8B ZS"][i],
  }));

  const [data, setData] = useState(buildData([null, null, null, null]));

  useEffect(() => {
    async function load() {
      if (!agentName) return;
      const scores = await Promise.all(tasks.map(async (t) => {
        const log = await fetchJSON(`${BASE_URL}/training_log?agent_name=${encodeURIComponent(agentName)}`);
        if (!log?.entries) return null;
        const taskEntries = log.entries.filter(e => e.task_id === t);
        if (!taskEntries.length) return null;
        return taskEntries[taskEntries.length - 1].grader_score;
      }));
      setData(buildData(scores));
    }
    load();
  }, [agentName]);

  return (
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <span style={styles.cardTitle}>Task Comparison</span>
        {agentName && <span style={{ ...styles.badge, background: "#1a2e1a" }}>{agentName}</span>}
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={COLORS.border} />
          <XAxis dataKey="task" stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} />
          <YAxis domain={[0, 1]} stroke={COLORS.muted} tick={{ fill: COLORS.muted, fontSize: 11 }} />
          <Tooltip
            contentStyle={{ backgroundColor: COLORS.card, border: `1px solid ${COLORS.border}`, color: COLORS.text }}
            formatter={(v) => v != null ? v.toFixed(4) : "—"}
          />
          <Legend wrapperStyle={{ color: COLORS.muted, fontSize: 11 }} />
          <Bar dataKey="score" fill={COLORS.blue} name="Your Agent" radius={[3, 3, 0, 0]} />
          <Bar dataKey="allAllow" fill={COLORS.gray} name="All-Allow" radius={[3, 3, 0, 0]} />
          <Bar dataKey="llama8b" fill={COLORS.orange} name="Llama-8B ZS" radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// 4. Task 4 Adversary State Heatmap (topic × intensity)
function AdversaryStatePanel({ sessionId }) {
  const [trajectory, setTrajectory] = useState([]);

  useEffect(() => {
    async function load() {
      if (!sessionId) return;
      const data = await fetchJSON(`${BASE_URL}/adversary_state?session_id=${sessionId}`);
      if (data?.adversary_trajectory) setTrajectory(data.adversary_trajectory);
    }
    load();
  }, [sessionId]);

  if (!trajectory.length) {
    return (
      <div style={styles.card}>
        <div style={styles.cardHeader}>
          <span style={styles.cardTitle}>Task 4: Adversary Trajectory</span>
        </div>
        <div style={styles.empty}>Complete a Task 4 episode to see adversary trajectory.</div>
      </div>
    );
  }

  const intensityColors = ["#22c55e", "#86efac", "#fbbf24", "#f97316", "#ef4444", "#7c3aed"];
  const intensityLabels = ["Safe (0)", "Low (1)", "Med (2)", "Elev (3)", "High (4)", "Crit (5)"];

  return (
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <span style={styles.cardTitle}>Task 4: Adversary Trajectory</span>
        <span style={{ ...styles.badge, background: "#2d1b69" }}>
          Topics: {new Set(trajectory.map(t => t.topic_idx)).size}
        </span>
      </div>
      <div style={{ overflowX: "auto" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 11 }}>
          <thead>
            <tr>
              {["Turn", "Topic", "Intensity", "Agent Action", "Correct?"].map(h => (
                <th key={h} style={{ padding: "6px 8px", color: COLORS.muted, textAlign: "left",
                                     borderBottom: `1px solid ${COLORS.border}` }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {trajectory.map((step, i) => (
              <tr key={i} style={{ borderBottom: `1px solid ${COLORS.border}` }}>
                <td style={{ padding: "5px 8px", color: COLORS.text }}>{step.turn}</td>
                <td style={{ padding: "5px 8px", color: COLORS.text }}>Topic {step.topic_idx}</td>
                <td style={{ padding: "5px 8px" }}>
                  <span style={{
                    display: "inline-block", padding: "2px 6px", borderRadius: 4,
                    background: intensityColors[step.intensity] + "33",
                    color: intensityColors[step.intensity],
                    fontSize: 10, fontWeight: "bold",
                  }}>
                    {intensityLabels[step.intensity]}
                  </span>
                </td>
                <td style={{ padding: "5px 8px" }}>
                  <span style={{
                    display: "inline-block", padding: "2px 6px", borderRadius: 4,
                    background: step.agent_action === "allow" ? "#1a2e1a" : "#2d1b1b",
                    color: step.agent_action === "allow" ? COLORS.green : COLORS.red,
                    fontSize: 10,
                  }}>
                    {step.agent_action}
                  </span>
                </td>
                <td style={{ padding: "5px 8px", color: step.correct ? COLORS.green : COLORS.red }}>
                  {step.correct ? "✓" : "✗"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// 5. Agent Leaderboard
function LeaderboardPanel() {
  const [entries, setEntries] = useState([]);
  const [task, setTask] = useState("basic_threat_detection");

  useEffect(() => {
    async function load() {
      const data = await fetchJSON(`${BASE_URL}/leaderboard`);
      if (data?.leaderboard) {
        const taskEntries = data.leaderboard.filter(e => !task || e.task_id === task);
        setEntries(taskEntries.slice(0, 10));
      }
    }
    load();
    const interval = setInterval(load, REFRESH_INTERVAL_MS * 2);
    return () => clearInterval(interval);
  }, [task]);

  return (
    <div style={styles.card}>
      <div style={styles.cardHeader}>
        <span style={styles.cardTitle}>Leaderboard</span>
        <select style={styles.select} value={task} onChange={e => setTask(e.target.value)}>
          <option value="basic_threat_detection">Task 1</option>
          <option value="context_aware_policy">Task 2</option>
          <option value="multiturn_adversarial">Task 3</option>
          <option value="adversarial_adaptation">Task 4</option>
        </select>
      </div>
      {entries.length === 0 ? (
        <div style={styles.empty}>No leaderboard entries yet.</div>
      ) : (
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
          <thead>
            <tr>
              {["Rank", "Agent", "Score"].map(h => (
                <th key={h} style={{ padding: "6px 8px", color: COLORS.muted, textAlign: "left",
                                     borderBottom: `1px solid ${COLORS.border}` }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {entries.map((e, i) => (
              <tr key={i} style={{ borderBottom: `1px solid ${COLORS.border}` }}>
                <td style={{ padding: "5px 8px", color: COLORS.muted, fontWeight: "bold" }}>#{i + 1}</td>
                <td style={{ padding: "5px 8px", color: COLORS.text }}>{e.agent_name || "—"}</td>
                <td style={{ padding: "5px 8px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                    <div style={{
                      width: `${(e.score ?? 0) * 80}px`, height: 8,
                      background: COLORS.blue, borderRadius: 4, minWidth: 4,
                    }} />
                    <span style={{ color: COLORS.text, fontWeight: "bold" }}>
                      {(e.score ?? 0).toFixed(4)}
                    </span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

// ── Task 4 Episode Runner ─────────────────────────────────────────────────────

async function runTask4Episode() {
  const resetRes = await fetchJSON(`${BASE_URL}/reset?task_id=adversarial_adaptation`);
  if (!resetRes?.session_id) throw new Error("Reset failed");
  const sid = resetRes.session_id;
  let obs = resetRes;
  let done = false;
  let steps = 0;
  while (!done && steps < 15) {
    const action = {
      prompt_id: obs.prompt_id,
      action_type: obs.user_risk_score > 0.5 ? "refuse" : "allow",
      reason: "dashboard demo",
      modified_prompt: null,
    };
    const res = await fetch(`${BASE_URL}/step?session_id=${sid}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(action),
      signal: AbortSignal.timeout(15000),
    });
    if (!res.ok) break;
    const data = await res.json();
    done = data.done;
    obs = data.observation || obs;
    steps++;
  }
  return { sessionId: sid, done };
}

// ── Main Dashboard ────────────────────────────────────────────────────────────

export default function TrainingDashboard() {
  const [urlInput, setUrlInput] = useState(BASE_URL);
  const [agentName, setAgentName] = useState("guardrail_trl_agent");
  const [sessionId, setSessionId] = useState("");
  const [task4SessionId, setTask4SessionId] = useState("");
  const [envStatus, setEnvStatus] = useState("checking");
  const [, setMultiAgentInfo] = useState(null);
  const [lastRefresh, setLastRefresh] = useState(new Date());
  const [task4Running, setTask4Running] = useState(false);
  const [task4Msg, setTask4Msg] = useState("");

  function handleConnect() {
    BASE_URL = urlInput.replace(/\/+$/, "");
    setEnvStatus("checking");
    setMultiAgentInfo(null);
  }

  async function handleRunTask4() {
    setTask4Running(true);
    setTask4Msg("Running Task 4 episode...");
    try {
      const { sessionId: sid, done } = await runTask4Episode();
      setTask4SessionId(sid);
      setTask4Msg(done
        ? `Episode complete. Session: ${sid.slice(0, 8)}...`
        : `Episode partially complete (${sid.slice(0, 8)}...) — trajectory may be incomplete`);
    } catch (e) {
      setTask4Msg(`Error: ${e.message}`);
    } finally {
      setTask4Running(false);
    }
  }

  useEffect(() => {
    async function checkHealth() {
      const r = await fetchJSON(`${BASE_URL}/health`);
      setEnvStatus(r ? "online" : "offline");
    }
    async function loadInfo() {
      const info = await fetchJSON(`${BASE_URL}/multi_agent_info`);
      if (info) setMultiAgentInfo(info);
    }
    checkHealth();
    loadInfo();
    const interval = setInterval(() => {
      checkHealth();
      setLastRefresh(new Date());
    }, REFRESH_INTERVAL_MS);
    return () => clearInterval(interval);
  }, []);

  return (
    <div style={styles.root}>
      {/* URL Input + Status Row */}
      <div style={styles.urlBar}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%", flexShrink: 0,
            background: envStatus === "online" ? COLORS.green : envStatus === "offline" ? COLORS.red : COLORS.yellow,
          }} />
          <span style={{ color: COLORS.muted, fontSize: 11, minWidth: 80 }}>
            {envStatus === "online" ? "Online" : envStatus === "offline" ? "Offline" : "Checking..."}
          </span>
        </div>
        <input
          style={{ ...styles.input, flex: 1, maxWidth: 500 }}
          value={urlInput}
          onChange={e => setUrlInput(e.target.value)}
          onKeyDown={e => e.key === "Enter" && handleConnect()}
          placeholder="https://varunventra-guardrail-arena.hf.space"
        />
        <button style={styles.btnPrimary} onClick={handleConnect}>Connect</button>
        <span style={{ color: COLORS.muted, fontSize: 11, marginLeft: 8 }}>
          {lastRefresh.toLocaleTimeString()}
        </span>
      </div>

      {/* Header */}
      <div style={styles.header}>
        <div>
          <h1 style={styles.title}>Guardrail Arena</h1>
          <p style={styles.subtitle}>Training Dashboard — Multi-Agent Interactions (Theme #1)</p>
        </div>
      </div>

      {/* Key Numbers Strip */}
      <div style={styles.infoBar}>
        <div style={styles.infoItem}>
          <span style={styles.infoLabel}>Qwen-235B on Task 4</span>
          <span style={{ ...styles.infoValue, color: COLORS.red }}>0.0000</span>
        </div>
        <div style={styles.infoItem}>
          <span style={styles.infoLabel}>Q-learner on Task 4</span>
          <span style={{ ...styles.infoValue, color: COLORS.green }}>0.9540</span>
        </div>
        <div style={styles.infoItem}>
          <span style={styles.infoLabel}>Branching Convos</span>
          <span style={styles.infoValue}>30</span>
        </div>
        <div style={styles.infoItem}>
          <span style={styles.infoLabel}>Adversary States</span>
          <span style={styles.infoValue}>180</span>
        </div>
        <div style={styles.infoItem}>
          <span style={styles.infoLabel}>Llama-8B ZS (Task 1)</span>
          <span style={styles.infoValue}>0.5428</span>
        </div>
      </div>

      {/* Multi-Agent Framing Panel */}
      <div style={{ ...styles.card, marginBottom: 16 }}>
        <div style={styles.cardHeader}>
          <span style={styles.cardTitle}>Multi-Agent Architecture</span>
          <span style={{ ...styles.badge, background: "#1a1a2e", color: COLORS.purple }}>Theme #1: Multi-Agent Interactions</span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>
          <div style={{ background: "#0d1117", borderRadius: 6, padding: "12px 16px", border: `1px solid ${COLORS.border}` }}>
            <div style={{ color: COLORS.red, fontWeight: 700, marginBottom: 8 }}>Adversary: DeterministicAdversary FSM</div>
            <div style={{ color: COLORS.muted, fontSize: 12, lineHeight: 1.6 }}>
              60 states (10 topics × 6 intensities) × 3 surface variants = 180 observable states<br/>
              Adapts in real-time to defender's actions<br/>
              <span style={{ color: COLORS.orange }}>allow → escalate intensity | block → back off</span>
            </div>
          </div>
          <div style={{ background: "#0d1117", borderRadius: 6, padding: "12px 16px", border: `1px solid ${COLORS.border}` }}>
            <div style={{ color: COLORS.green, fontWeight: 700, marginBottom: 8 }}>Defender: Trainable Safety Policy</div>
            <div style={{ color: COLORS.muted, fontSize: 12, lineHeight: 1.6 }}>
              Observation: prompt + context + risk_score<br/>
              Action: allow / refuse / modify / escalate<br/>
              <span style={{ color: COLORS.blue }}>Reward: +0.20 correct block | -0.30 missed attack</span>
            </div>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div style={styles.controls}>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Agent Name</label>
          <input
            style={styles.input}
            value={agentName}
            onChange={e => setAgentName(e.target.value)}
            placeholder="Agent name from training_log"
          />
        </div>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Session ID (for breakdown)</label>
          <input
            style={styles.input}
            value={sessionId}
            onChange={e => setSessionId(e.target.value)}
            placeholder="UUID from /reset"
          />
        </div>
        <div style={styles.inputGroup}>
          <label style={styles.label}>Task 4 Demo</label>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button
              style={{ ...styles.btnPrimary, opacity: task4Running ? 0.6 : 1 }}
              onClick={handleRunTask4}
              disabled={task4Running}
            >
              {task4Running ? "Running..." : "Run Task 4 Episode"}
            </button>
            {task4Msg && <span style={{ fontSize: 11, color: COLORS.muted }}>{task4Msg}</span>}
          </div>
        </div>
      </div>

      {/* Main grid */}
      <div style={styles.grid}>
        <RewardCurvePanel agentName={agentName} />
        <ActionDistributionPanel sessionId={sessionId} />
        <TaskComparisonPanel agentName={agentName} />
        {task4SessionId && <AdversaryStatePanel sessionId={task4SessionId} />}
        <div style={{ gridColumn: task4SessionId ? "span 1" : "span 2" }}>
          <LeaderboardPanel />
        </div>
      </div>

      {/* Footer */}
      <div style={styles.footer}>
        <span>Guardrail Arena — Meta × HuggingFace × Cerebral Valley OpenEnv Hackathon</span>
        <a
          href={`${BASE_URL}/docs`}
          target="_blank"
          rel="noopener noreferrer"
          style={{ color: COLORS.blue }}
        >
          API Docs
        </a>
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
    padding: 20,
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "flex-start",
    marginBottom: 16,
    paddingBottom: 16,
    borderBottom: `1px solid ${COLORS.border}`,
  },
  title: {
    margin: 0,
    fontSize: 22,
    fontWeight: 700,
    color: COLORS.text,
  },
  subtitle: {
    margin: "4px 0 0",
    fontSize: 13,
    color: COLORS.muted,
  },
  infoBar: {
    display: "flex",
    gap: 24,
    padding: "10px 16px",
    background: COLORS.card,
    border: `1px solid ${COLORS.border}`,
    borderRadius: 8,
    marginBottom: 16,
    flexWrap: "wrap",
  },
  infoItem: {
    display: "flex",
    flexDirection: "column",
    gap: 2,
  },
  infoLabel: {
    fontSize: 10,
    color: COLORS.muted,
    textTransform: "uppercase",
    letterSpacing: "0.05em",
  },
  infoValue: {
    fontSize: 18,
    fontWeight: 700,
    color: COLORS.text,
  },
  controls: {
    display: "flex",
    gap: 16,
    marginBottom: 16,
    flexWrap: "wrap",
  },
  inputGroup: {
    display: "flex",
    flexDirection: "column",
    gap: 4,
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
    padding: "6px 10px",
    fontSize: 13,
    outline: "none",
    minWidth: 220,
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
  },
  footer: {
    marginTop: 20,
    paddingTop: 16,
    borderTop: `1px solid ${COLORS.border}`,
    display: "flex",
    justifyContent: "space-between",
    fontSize: 11,
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
  btnPrimary: {
    background: COLORS.blue,
    border: "none",
    borderRadius: 6,
    color: "#fff",
    padding: "6px 14px",
    fontSize: 12,
    fontWeight: 600,
    cursor: "pointer",
    whiteSpace: "nowrap",
  },
};
