#!/usr/bin/env bash
# Run from project root (locally with ADC, or on the GCP VM).
# Step 1: materialize the joined table in BigQuery.
# Step 2: export to GCS as sharded parquet for training.
#
# Before running, confirm signal table names:
#   bq ls gh-issue-ml-2026-491320:issues

set -euo pipefail

PROJECT=gh-issue-ml-2026-491320
DATASET=issues
OUTPUT_TABLE=${PROJECT}:${DATASET}.issues_with_signals
GCS_DEST="gs://gh_issue_ml-data/issues/issues_with_signals/*.parquet"

echo "==> Materializing ${OUTPUT_TABLE} ..."
bq query \
  --use_legacy_sql=false \
  --project_id="${PROJECT}" \
  < sql/create_issues_with_signals.sql

echo "==> Exporting to ${GCS_DEST} ..."
bq extract \
  --destination_format=PARQUET \
  --project_id="${PROJECT}" \
  "${OUTPUT_TABLE}" \
  "${GCS_DEST}"

echo "==> Done."
