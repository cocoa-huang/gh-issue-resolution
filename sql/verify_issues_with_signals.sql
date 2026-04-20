-- Verification queries — run after export_issues_with_signals.sh completes.
-- All three checks should pass before training Model 2.

-- Check 1: Row count must match source exactly (LEFT JOINs preserve all issues).
SELECT
  (SELECT COUNT(*) FROM `gh-issue-ml-2026-491320.issues.issues_labeled_2025`) AS source_rows,
  (SELECT COUNT(*) FROM `gh-issue-ml-2026-491320.issues.issues_with_signals`)  AS joined_rows;

-- Check 2: Spot-check one known active repo.
-- Compares joined value against a manual re-aggregation from the raw signal table.
-- Both columns should agree. If they differ, the join predicate is wrong.
SELECT
  w.issue_created_at,
  w.pr_merged_30d                                           AS joined_value,
  (
    SELECT COALESCE(SUM(p.pr_merged_count), 0)
    FROM `gh-issue-ml-2026-491320.issues.pr_daily_2025` p
    WHERE p.repo_name = w.repo_name
      AND p.date >= DATE_SUB(DATE(TIMESTAMP(w.issue_created_at)), INTERVAL 30 DAY)
      AND p.date  < DATE(TIMESTAMP(w.issue_created_at))
  )                                                         AS manual_recompute
FROM `gh-issue-ml-2026-491320.issues.issues_with_signals` w
WHERE w.repo_name = 'microsoft/vscode'
ORDER BY w.issue_created_at
LIMIT 5;

-- Check 3: Signal distributions.
-- Red flag: >95% zeros for any signal suggests a repo_name format mismatch
-- (e.g. "owner/repo" in one table vs "owner%2Frepo" in another).
SELECT
  ROUND(COUNTIF(pr_merged_30d     = 0) / COUNT(*), 3) AS frac_zero_pr,
  ROUND(COUNTIF(push_count_30d    = 0) / COUNT(*), 3) AS frac_zero_push,
  ROUND(COUNTIF(release_count_90d = 0) / COUNT(*), 3) AS frac_zero_release,
  ROUND(COUNTIF(star_count_30d    = 0) / COUNT(*), 3) AS frac_zero_stars,
  ROUND(AVG(pr_merged_30d),    1) AS avg_pr_merged,
  ROUND(AVG(push_count_30d),   1) AS avg_push_count,
  ROUND(AVG(release_count_90d),1) AS avg_releases,
  ROUND(AVG(star_count_30d),   1) AS avg_stars
FROM `gh-issue-ml-2026-491320.issues.issues_with_signals`;
