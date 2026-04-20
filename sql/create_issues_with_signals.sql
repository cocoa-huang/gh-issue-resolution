-- Model 2 data prep: temporal join of issues with repo signals.
--
-- Output table: issues.issues_with_signals
-- Rows: one per issue (same count as issues_labeled_2025 — LEFT JOINs preserve all)
-- New columns: pr_merged_30d, avg_merge_hours_30d, push_count_30d,
--              release_count_90d, star_count_30d
--
-- Temporal boundary: all signal windows end strictly before issue_created_at.
-- Run via: bash sql/export_issues_with_signals.sh

CREATE OR REPLACE TABLE `gh-issue-ml-2026-491320.issues.issues_with_signals` AS

WITH issues AS (
  SELECT
    repo_name,
    title,
    body,
    author_association,
    issue_created_at,
    label,
    DATE(TIMESTAMP(issue_created_at)) AS issue_date
  FROM `gh-issue-ml-2026-491320.issues.issues_labeled_2025`
),

pr_agg AS (
  SELECT
    i.repo_name,
    i.issue_created_at,
    COALESCE(SUM(p.pr_merged_count), 0) AS pr_merged_30d,
    AVG(p.avg_hours_to_merge)            AS avg_merge_hours_30d
  FROM issues i
  LEFT JOIN `gh-issue-ml-2026-491320.issues.pr_daily_2025` p
    ON  p.repo_name = i.repo_name
    AND p.date >= DATE_SUB(i.issue_date, INTERVAL 30 DAY)
    AND p.date  < i.issue_date
  GROUP BY i.repo_name, i.issue_created_at
),

push_agg AS (
  SELECT
    i.repo_name,
    i.issue_created_at,
    COALESCE(SUM(p.push_count), 0) AS push_count_30d
  FROM issues i
  LEFT JOIN `gh-issue-ml-2026-491320.issues.push_daily_2025` p
    ON  p.repo_name = i.repo_name
    AND p.date >= DATE_SUB(i.issue_date, INTERVAL 30 DAY)
    AND p.date  < i.issue_date
  GROUP BY i.repo_name, i.issue_created_at
),

release_agg AS (
  SELECT
    i.repo_name,
    i.issue_created_at,
    COALESCE(SUM(r.release_count), 0) AS release_count_90d
  FROM issues i
  LEFT JOIN `gh-issue-ml-2026-491320.issues.release_daily_2025` r
    ON  r.repo_name = i.repo_name
    AND r.date >= DATE_SUB(i.issue_date, INTERVAL 90 DAY)
    AND r.date  < i.issue_date
  GROUP BY i.repo_name, i.issue_created_at
),

watch_agg AS (
  SELECT
    i.repo_name,
    i.issue_created_at,
    COALESCE(SUM(w.star_count), 0) AS star_count_30d
  FROM issues i
  LEFT JOIN `gh-issue-ml-2026-491320.issues.watch_daily_2025` w
    ON  w.repo_name = i.repo_name
    AND w.date >= DATE_SUB(i.issue_date, INTERVAL 30 DAY)
    AND w.date  < i.issue_date
  GROUP BY i.repo_name, i.issue_created_at
)

SELECT
  i.repo_name,
  i.title,
  i.body,
  i.author_association,
  i.issue_created_at,
  i.label,
  COALESCE(pr.pr_merged_30d,       0) AS pr_merged_30d,
  pr.avg_merge_hours_30d,
  COALESCE(push.push_count_30d,    0) AS push_count_30d,
  COALESCE(rel.release_count_90d,  0) AS release_count_90d,
  COALESCE(watch.star_count_30d,   0) AS star_count_30d
FROM issues i
LEFT JOIN pr_agg      pr    USING (repo_name, issue_created_at)
LEFT JOIN push_agg    push  USING (repo_name, issue_created_at)
LEFT JOIN release_agg rel   USING (repo_name, issue_created_at)
LEFT JOIN watch_agg   watch USING (repo_name, issue_created_at)
