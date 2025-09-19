# pip install numpy pandas scipy
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# ==================== CONFIG ====================
PREFS_CSV = "prefs_2025fall.csv"        # wide format with columns: student, rank1..rank5
TOPICS_CSV = "topics.csv"          # optional: master list of topics (not currently used)
SAVE_PREFIX = "assignment_2025fall"      # writes assignment.csv, assignment_summary.csv
SEED = 2025                     # change for a different random outcome; None -> non-deterministic
EPS_JITTER = 1e-3               # tiny noise to break ties without changing rank order

# RANK_TO_COST: cost associated with each rank (lower = better) {Rank1: Cost1, ...}. Can include more ranks if desired (e.g., rank6..rank10)
# Strictly convex penalties (bigger jump each step). Keeps 1st >> 2nd > 3rd... while staying small numerically
RANK_TO_COST = {1: 0, 
                2: 2, 
                3: 5, 
                4: 9, 
                5: 17}

# Large penalty for topics a student did not list (still allows assignment)
PENALTY_UNLISTED = 1000
# =================================================

rng = np.random.default_rng(SEED)

# Print config
print("Running assignment with config:")
print(f"PREFS_CSV = {PREFS_CSV}\n SAVE_PREFIX = {SAVE_PREFIX}\n SEED = {SEED}\n EPS_JITTER = {EPS_JITTER}\n RANK_TO_COST = {RANK_TO_COST}\n PENALTY_UNLISTED = {PENALTY_UNLISTED}\n")

def load_wide_csv(path: str):
    df = pd.read_csv(path, comment="#").fillna("")
    if "student" not in df.columns:
        raise ValueError("CSV must include a 'student' column.")
    # normalize to strings
    df["student"] = df["student"].astype(str)
    # detect rank columns in order
    rank_cols = [c for c in df.columns if c.lower().startswith("rank")]
    if not rank_cols:
        raise ValueError("No rank columns found (expected rank1..rank5).")
    # sort rank columns by their numeric suffix if present
    def rank_key(c):
        try:
            return int(''.join(ch for ch in c if ch.isdigit()))
        except ValueError:
            return 10**9
    rank_cols = sorted(rank_cols, key=rank_key)

    # canonicalize topics: strip spaces, unify case if desired (here: keep as-is except strip)
    for c in rank_cols:
        df[c] = df[c].astype(str).str.strip()

    students = df["student"].tolist()

    # Always load topics from TOPICS_CSV
    topics_all = pd.read_csv(TOPICS_CSV, header=None).squeeze().astype(str).str.strip().tolist()
    # Topic universe: use any topic that appears at least once
    topics_prefs = sorted({t for c in rank_cols for t in df[c].tolist() if t})
    # Check if all topics in prefs are in the master topic list
    missing_topics = sorted(set(topics_prefs) - set(topics_all))
    if missing_topics:
        print(f"WARNING: The following topics appear in preferences but not in {TOPICS_CSV}: {missing_topics}. Taking the union.")
    # Take the union of both lists for the final topic universe
    topics_union = sorted(set(topics_all).union(topics_prefs))
    return df, students, topics_union, rank_cols

def dedup_keep_highest_rank(df: pd.DataFrame, rank_cols):
    """
    For each student, if a topic is listed multiple times, keep the highest preference (lowest rank number).
    Implementation detail: we build a mapping topic -> best rank index per student.
    """
    cleaned = []
    for _, row in df.iterrows():
        seen_best_rank_idx = {}  # topic -> best (lowest) rank index (1-based)
        for r_idx, col in enumerate(rank_cols, start=1):
            t = row[col].strip()
            if not t:
                continue
            if t not in seen_best_rank_idx or r_idx < seen_best_rank_idx[t]:
                seen_best_rank_idx[t] = r_idx
        # Rebuild a single row with topics placed in their (unique) best rank positions
        new_row = {"student": row["student"], **{c: "" for c in rank_cols}}
        # Place topics by their best index; if clashes (rare), keep first and push the other later
        # But since we computed minimum index per topic, there are no duplicates now.
        for t, best_idx in seen_best_rank_idx.items():
            new_row[rank_cols[best_idx - 1]] = t
        cleaned.append(new_row)
    return pd.DataFrame(cleaned, columns=["student"] + rank_cols)

def build_cost_matrix(df, students, topics, rank_cols):
    nS, nT = len(students), len(topics)
    cost = np.full((nS, nT), PENALTY_UNLISTED, dtype=float)
    topic_index = {t: j for j, t in enumerate(topics)}
    # Fill costs using best rank only (duplication already removed)
    student_index = {s: i for i, s in enumerate(students)}
    for _, row in df.iterrows():
        i = student_index[row["student"]]
        for r, col in enumerate(rank_cols, start=1):
            t = row[col].strip()
            if t and t in topic_index:
                cost[i, topic_index[t]] = min(cost[i, topic_index[t]], RANK_TO_COST[r])
    return cost

def randomize_and_jitter(cost, rng, eps=1e-6):
    nS, nT = cost.shape
    row_perm = rng.permutation(nS)
    col_perm = rng.permutation(nT)
    cost_shuf = cost[row_perm][:, col_perm].copy()
    if eps and eps > 0:
        cost_shuf += rng.uniform(0.0, eps, size=cost_shuf.shape)
    return cost_shuf, row_perm, col_perm

def inverse_perm(perm):
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    return inv

def first_choice_lottery(df_clean: pd.DataFrame, students, topics, rank_cols, rng):
    """
    Returns:
      winners: list of (student, topic) fixed assignments (rank = 1)
      remaining_students: list[str]
      remaining_topics: list[str]
      df_remaining: df with remaining students only
    Logic:
      - For each topic, collect all students whose rank1 == topic.
      - If exactly one: that student wins the topic.
      - If >1: pick one uniformly at random; others remain unassigned for optimization.
    """
    rank1_col = rank_cols[0]
    # group students by their first choice topic
    topic_to_candidates = {}
    for s, t in zip(df_clean["student"], df_clean[rank1_col]):
        if t:
            topic_to_candidates.setdefault(t, []).append(s)

    winners = []
    taken_students = set()
    taken_topics = set()

    for t, cand_students in topic_to_candidates.items():
        if t not in topics:
            continue
        if len(cand_students) == 1:
            s = cand_students[0]
            winners.append((s, t))
            taken_students.add(s)
            taken_topics.add(t)
        elif len(cand_students) > 1:
            # lottery among candidates
            s = rng.choice(cand_students)
            winners.append((s, t))
            taken_students.add(s)
            taken_topics.add(t)
            # non-winners from this group will remain for optimization with their other ranks

    remaining_students = [s for s in students if s not in taken_students]
    remaining_topics = [t for t in topics if t not in taken_topics]
    df_remaining = df_clean[df_clean["student"].isin(remaining_students)].copy()

    return winners, remaining_students, remaining_topics, df_remaining

def main():
    df_raw, students_all, topics_all, rank_cols = load_wide_csv(PREFS_CSV)

    # De-duplicate per student (keep highest-ranked instance of any repeated topic)
    print("Cleaning duplicates...")
    df = dedup_keep_highest_rank(df_raw, rank_cols)

    # First-choice lottery pre-step
    print("Running first-choice lottery...")
    winners, students_rem, topics_rem, df_rem = first_choice_lottery(df, students_all, topics_all, rank_cols, rng)

    # If everyone got their #1 (rare with many topics), we're done
    if len(students_rem) == 0:
        assigned = pd.DataFrame({
            "student": [s for s, _ in winners],
            "topic_id": [t for _, t in winners],
            "cost": [RANK_TO_COST[1]] * len(winners),
            "rank_assigned": [1] * len(winners)
        }).sort_values("student")
    else:
        # Build and solve for the remaining set
        # Guarantee we still have enough topics (problem statement says topics > students overall)
        if len(topics_rem) < len(students_rem):
            raise ValueError(
                f"Not enough remaining topics ({len(topics_rem)}) for remaining students ({len(students_rem)}). "
                "Check inputs or provide a master topic list."
            )
        
        print(f"Assigning remaining {len(students_rem)} students to {len(topics_rem)} topics...")
        cost = build_cost_matrix(df_rem, students_rem, topics_rem, rank_cols)

        # Random tie-breaking for the optimization step
        cost_shuf, row_perm, col_perm = randomize_and_jitter(cost, rng, EPS_JITTER)
        r_idx_shuf, c_idx_shuf = linear_sum_assignment(cost_shuf)
        r_idx = row_perm[r_idx_shuf]
        c_idx = col_perm[c_idx_shuf]

        # Build assignment for remaining
        assigned_rem = pd.DataFrame({
            "student": [students_rem[i] for i in r_idx],
            "topic_id": [topics_rem[j] for j in c_idx],
            "cost": [cost[i, j] for i, j in zip(r_idx, c_idx)]
        })

        inv_cost_to_rank = {v: k for k, v in RANK_TO_COST.items()}
        assigned_rem["rank_assigned"] = assigned_rem["cost"].map(lambda c: inv_cost_to_rank.get(c, None))

        # Combine with winners (rank 1)
        assigned = pd.concat([
            pd.DataFrame({
                "student": [s for s, _ in winners],
                "topic_id": [t for _, t in winners],
                "cost": [RANK_TO_COST[1]] * len(winners),
                "rank_assigned": [1] * len(winners)
            }),
            assigned_rem
        ], ignore_index=True).sort_values("student")
    
    # Add columns: 'won_lottery' and 'assigned_unranked'
    lottery_winners = set(s for s, _ in winners)
    assigned["won_lottery"] = assigned["student"].apply(lambda s: s in lottery_winners)
    assigned["assigned_unranked"] = assigned["rank_assigned"].isna()

    # Sanity checks
    assert assigned["student"].nunique() == len(students_all), "Some students not assigned."
    assert assigned["topic_id"].nunique() == len(assigned), "A topic was assigned to multiple students."

    # Metrics
    summary = {
        "n_students": len(students_all),
        "n_topics": len(topics_all),
        "won_1st_choice_lottery": len(winners),
        "got_1st_choice": int((assigned["rank_assigned"] == 1).sum()),
        "top2": int(assigned["rank_assigned"].isin([1, 2]).sum()),
        "top3": int(assigned["rank_assigned"].isin([1, 2, 3]).sum()),
        "avg_rank_among_ranked": float(assigned["rank_assigned"].dropna().mean())
            if (~assigned["rank_assigned"].isna()).any() else None,
        "unranked_assigned": int(assigned["rank_assigned"].isna().sum())
    }
    summary_df = pd.DataFrame([summary])

    # Save
    assigned.to_csv(f"{SAVE_PREFIX}.csv", index=False)
    summary_df.to_csv(f"{SAVE_PREFIX}_summary.csv", index=False)
    
    print("Summary statistics:")
    print(summary_df.to_string(index=False))
    if SEED is not None:
        print(f"\nRandomness is reproducible with SEED={SEED}.")

if __name__ == "__main__":
    main()
