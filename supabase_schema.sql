-- Run in Supabase SQL editor (Dashboard → SQL).
-- Service role key (backend only) bypasses RLS; still enable RLS for defense in depth.

create table if not exists public.taxonomy_proposals (
  id text primary key,
  type text not null,
  status text not null default 'pending',
  created_at timestamptz not null default now(),
  bundle_version text not null default '',
  paragraph text not null default '',
  payload jsonb not null default '{}'::jsonb,
  rationale text not null default ''
);

alter table public.taxonomy_proposals enable row level security;

-- No public access via anon/authenticated keys (backend uses service_role only).
revoke all on public.taxonomy_proposals from anon, authenticated;

create index if not exists taxonomy_proposals_status_created_idx
  on public.taxonomy_proposals (status, created_at desc);

create table if not exists public.taxonomy_merge_log (
  id bigserial primary key,
  proposal_id text not null,
  merge_type text not null,
  created_at timestamptz not null default now(),
  bundle_version text not null default '',
  payload jsonb not null default '{}'::jsonb
);

alter table public.taxonomy_merge_log enable row level security;
revoke all on public.taxonomy_merge_log from anon, authenticated;
