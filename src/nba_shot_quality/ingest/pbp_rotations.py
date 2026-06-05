"""Reconstruct per-game player rotations from PlayByPlayV3 substitution events.

This is the fast alternative to the GameRotation endpoint: PlayByPlayV3 returns in ~0.2s/call
(vs ~13-30s for GameRotation). Pull SEQUENTIALLY with a small sleep — like the shot ingest — not
concurrently: nba_api's stats endpoints share an aggressive rate limiter, and even 6 workers
triggers heavy 429 throttling (measured: 6 workers => ~23% failures and ~7.8s/game amortized;
sequential + 0.6s sleep => ~0% failures and ~1s/game, i.e. a full two-season pull in ~40 min).

The trade-off vs GameRotation is that PBP gives substitution *events*, not check-in/out times — so
we walk the events to rebuild each player's per-period on-floor stints, inferring each period's
starting five from who is seen before being subbed in. Validated against GameRotation ground truth:
~96% of shots reconstruct to exactly 5v5 and, where they do, the on-floor 10 match GameRotation
99% of the time. The downstream 5v5 assertion in `shot_lineups` drops the ~4% that don't.

Output schema is identical to `ingest/rotations.py` (the GameRotation path) so the rest of the
RAPM pipeline (`shot_lineups`, `rapm`) consumes it unchanged: one row per stint with
`game_id, team_id, person_id, player_name, in_time, out_time`, where times are tenths of a second
of elapsed game time — the same scale `shot_lineups._elapsed_tenths` produces for shots.
"""

from __future__ import annotations

import re
import time
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import playbyplayv3
from tenacity import retry, stop_after_attempt, wait_exponential

REPO_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
REQUEST_TIMEOUT_SEC = 30
REQUEST_SLEEP_SEC = 0.6  # sequential pacing; concurrency triggers nba_api 429 throttling (see docstring)

KEEP_COLS = ["game_id", "team_id", "person_id", "player_name", "in_time", "out_time"]
REG_PERIOD_SEC = 720
OT_PERIOD_SEC = 300
SCALE = 10  # tenths of a second, matching shot_lineups._elapsed_tenths

_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")
_SUB_RE = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+)")
_INITIAL_RE = re.compile(r"^([A-Za-z]+)\.\s+(.+)$")  # "Jay. Williams" / "G. Antetokounmpo"


def _fold(s: str) -> str:
    """Lowercase + strip diacritics so 'Jokić' and 'Jokic' compare equal."""
    s = unicodedata.normalize("NFKD", str(s))
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


@lru_cache(maxsize=1)
def _static_first_last() -> dict[int, tuple[str, str]]:
    """person_id -> (folded first name, folded last name) from nba_api's static roster.

    PBP V3 has only the last name and a single initial, so sub descriptions that disambiguate
    same-initial teammates ('Jay. Williams' vs 'Jal. Williams') need real first names to resolve.
    """
    from nba_api.stats.static import players as _players

    return {int(p["id"]): (_fold(p["first_name"]), _fold(p["last_name"])) for p in _players.get_players()}


def _resolve_in_player(roster: dict[int, tuple[str, str]], desc_name: str) -> int | None:
    """Map a substitution's IN-player description name to a person_id within the team's roster."""
    m = _INITIAL_RE.match(desc_name.strip())
    prefix, last = (_fold(m.group(1)), _fold(m.group(2))) if m else (None, _fold(desc_name))
    cands = [pid for pid, (_first, last_name) in roster.items() if last_name == last]
    if len(cands) == 1:
        return cands[0]
    if prefix:  # disambiguate same-last-name teammates by first-name prefix (Jay -> Jaylin)
        narrowed = [pid for pid in cands if roster[pid][0].startswith(prefix)]
        if len(narrowed) == 1:
            return narrowed[0]
    return None


def _period_len_sec(period: int) -> int:
    return REG_PERIOD_SEC if period <= 4 else OT_PERIOD_SEC


def _period_start_tenths(period: int) -> int:
    """Elapsed game time (tenths) at the tip of a period."""
    regulation_done = min(period - 1, 4) * REG_PERIOD_SEC
    ot_done = max(period - 5, 0) * OT_PERIOD_SEC
    return (regulation_done + ot_done) * SCALE


def _elapsed_tenths(period: int, clock: str) -> int:
    """Elapsed game time (tenths) at a PBP event, from its period and 'PT06M43.00S' clock."""
    m = _CLOCK_RE.match(str(clock))
    remaining = float(m.group(1)) * 60 + float(m.group(2)) if m else 0.0
    elapsed_sec = _period_start_tenths(period) / SCALE + (_period_len_sec(period) - remaining)
    return int(round(elapsed_sec * SCALE))


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=8), reraise=True)
def _fetch_pbp(game_id: str) -> pd.DataFrame:
    return playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=REQUEST_TIMEOUT_SEC).get_data_frames()[0]


def reconstruct_stints(pbp: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """Walk a game's PBP events into on-floor stints (one row per continuous stint)."""
    df = pbp.copy()
    df["person_id"] = pd.to_numeric(df["personId"], errors="coerce").fillna(0).astype("int64")
    df["team_id"] = pd.to_numeric(df["teamId"], errors="coerce").fillna(0).astype("int64")
    df["period"] = df["period"].astype(int)
    df["elapsed"] = [_elapsed_tenths(p, c) for p, c in zip(df["period"], df["clock"])]
    df["is_sub"] = df["actionType"].astype(str).str.lower().str.contains("sub")

    # Maps: id -> display name, id -> team, and a per-team roster (id -> folded first/last names)
    # for resolving the IN player named in each substitution description.
    valid = df[df["person_id"] > 0]
    id_to_name = dict(zip(valid["person_id"], valid["playerName"].astype(str)))
    id_to_team: dict[int, int] = {}
    for pid, tid in zip(valid["person_id"], valid["team_id"]):
        if tid > 0:
            id_to_team.setdefault(int(pid), int(tid))
    static = _static_first_last()
    roster_by_team: dict[int, dict[int, tuple[str, str]]] = {}
    for pid, tid in id_to_team.items():
        # fall back to the event last name when a player is missing from the static roster
        first_last = static.get(pid, ("", _fold(id_to_name.get(pid, ""))))
        roster_by_team.setdefault(tid, {})[pid] = first_last

    # Parse subs into (elapsed, team_id, in_id, out_id), ordered by game time then action number.
    subs = []
    for _, r in df[df["is_sub"]].sort_values(["elapsed", "actionNumber"]).iterrows():
        out_id = int(r["person_id"])
        tid = int(r["team_id"])
        m = _SUB_RE.match(str(r["description"]))
        in_id = _resolve_in_player(roster_by_team.get(tid, {}), m.group(1)) if m else None
        if in_id is None or out_id <= 0 or tid <= 0:
            continue  # unresolved sub; the 5v5 guard downstream will drop affected shots
        id_to_team.setdefault(in_id, tid)
        subs.append((int(r["period"]), int(r["elapsed"]), tid, int(in_id), out_id))

    teams = sorted(t for t in df["team_id"].unique() if t > 0)
    if len(teams) != 2:
        return pd.DataFrame(columns=KEEP_COLS)

    # Reconstruct each period independently: the NBA PBP frequently omits the substitutions made
    # during a period break, so on-floor state cannot be carried across periods — instead re-derive
    # each period's starters from that period's own events (a player on the floor at the tip appears
    # NOT as a sub-in before any other involvement) and walk that period's subs.
    INF = float("inf")
    rows = []
    for p in sorted(df["period"].unique()):
        p = int(p)
        p_start = _period_start_tenths(p)
        p_end = p_start + _period_len_sec(p) * SCALE
        p_rows = df[(df["period"] == p) & (df["person_id"] > 0)]
        p_subs = [s for s in subs if s[0] == p]

        # First time each player is seen NOT as a sub-in (on the floor already), and first sub-in.
        t_present: dict[int, float] = {}
        for pid, e in zip(p_rows["person_id"].astype(int), p_rows["elapsed"].astype(float)):
            t_present[pid] = min(t_present.get(pid, INF), e)  # acting player / sub-OUT personId
        t_in: dict[int, float] = {}
        for _per, e, _tid, in_id, _out in p_subs:
            t_in[in_id] = min(t_in.get(in_id, INF), float(e))

        on_court: dict[int, set[int]] = {t: set() for t in teams}
        stint_start: dict[int, int] = {}
        for pid, tpres in t_present.items():
            if tpres < t_in.get(pid, INF):  # present before first sub-in => started the period
                tid = id_to_team.get(pid)
                if tid in on_court:
                    on_court[tid].add(pid)
                    stint_start[pid] = p_start

        for _per, e, tid, in_id, out_id in p_subs:
            if out_id in on_court.get(tid, set()):
                rows.append((tid, out_id, stint_start[out_id], e))
                on_court[tid].discard(out_id)
            if tid in on_court and in_id not in on_court[tid]:
                on_court[tid].add(in_id)
                stint_start[in_id] = e
        for tid in teams:
            for pid in on_court[tid]:
                rows.append((tid, pid, stint_start[pid], p_end))

    out = pd.DataFrame(rows, columns=["team_id", "person_id", "in_time", "out_time"])
    out = out[out["out_time"] > out["in_time"]]  # drop zero-length stints (sub in & out same tick)
    out["game_id"] = str(game_id).zfill(10)
    out["player_name"] = out["person_id"].map(id_to_name).fillna("")
    return out[KEEP_COLS]


def _pull_one(game_id: str, parts_dir: Path) -> str:
    """Fetch + reconstruct + cache one game's stint shard; return 'ok' | 'empty' | 'failed'."""
    try:
        pbp = _fetch_pbp(game_id)
    except Exception:
        return "failed"
    if pbp is None or pbp.empty:
        return "empty"
    stints = reconstruct_stints(pbp, game_id)
    if stints.empty:
        return "empty"
    stints.to_parquet(parts_dir / f"{game_id}.parquet", index=False)
    return "ok"


def ingest_pbp_rotations(season: str, force: bool = False) -> Path:
    """Reconstruct rotations for every game in the season from PlayByPlayV3 (sequential, resumable)."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / f"rotations_{season}.parquet"
    parts_dir = RAW_DIR / f"pbp_rotations_{season}_parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    scored_path = PROCESSED_DIR / f"shots_scored_{season}.parquet"
    if not scored_path.exists():
        raise FileNotFoundError(f"{scored_path} not found — run score --season {season} first")
    game_ids = sorted(pd.read_parquet(scored_path, columns=["game_id"])["game_id"].astype(str).unique())

    todo = [g for g in game_ids if force or not (parts_dir / f"{g}.parquet").exists()]
    print(f"[pbp_rotations] {len(game_ids):,} games for {season}: {len(todo):,} to pull, {len(game_ids)-len(todo):,} cached")

    # Sequential with a small sleep — concurrency triggers nba_api 429 throttling (see module docstring).
    counts = {"ok": 0, "empty": 0, "failed": 0}
    for done, g in enumerate(todo, 1):
        counts[_pull_one(g, parts_dir)] += 1
        time.sleep(REQUEST_SLEEP_SEC)
        if done % 100 == 0:
            print(f"  {done}/{len(todo)}  ok={counts['ok']} empty={counts['empty']} failed={counts['failed']}", flush=True)

    shards = sorted(parts_dir.glob("*.parquet"))
    all_rot = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    all_rot.to_parquet(out_path, index=False)
    missing = len(game_ids) - len(shards)
    print(f"[pbp_rotations] this run: ok={counts['ok']} empty={counts['empty']} failed={counts['failed']}")
    if missing:
        print(f"[pbp_rotations] {missing:,} games still missing — re-run to retry (resumes from shard cache)")
    print(f"[pbp_rotations] {len(all_rot):,} stint rows across {len(shards):,}/{len(game_ids):,} games -> {out_path}")
    return out_path
