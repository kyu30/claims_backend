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
  rationale text not null default '',
  reviewed_by text,
  reviewed_at timestamptz,
  applied_by text,
  applied_at timestamptz
);

alter table public.taxonomy_proposals enable row level security;

-- No public access via anon/authenticated keys (backend uses service_role only).
revoke all on public.taxonomy_proposals from anon, authenticated;

create index if not exists taxonomy_proposals_status_created_idx
  on public.taxonomy_proposals (status, created_at desc);

-- If you created the table before reviewer/apply attribution columns existed, run:
alter table public.taxonomy_proposals add column if not exists reviewed_by text;
alter table public.taxonomy_proposals add column if not exists reviewed_at timestamptz;
alter table public.taxonomy_proposals add column if not exists applied_by text;
alter table public.taxonomy_proposals add column if not exists applied_at timestamptz;

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

-- ============================================================================
-- Public taxonomy tables (browser read/write, NO auth)
--
-- WARNING: The policies below intentionally allow anonymous users to mutate
-- taxonomy. Only enable this for trusted/private deployments.
-- ============================================================================

create table if not exists public.taxonomy_superclaims (
  id text primary key,
  text text not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists public.taxonomy_subclaims (
  id text primary key,
  text text not null,
  superclaim_id text not null references public.taxonomy_superclaims(id) on update cascade on delete restrict,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create index if not exists taxonomy_subclaims_superclaim_idx
  on public.taxonomy_subclaims (superclaim_id);

alter table public.taxonomy_superclaims enable row level security;
alter table public.taxonomy_subclaims enable row level security;

-- Public read/write policies (anon + authenticated).
drop policy if exists "public superclaims read" on public.taxonomy_superclaims;
create policy "public superclaims read"
  on public.taxonomy_superclaims for select
  using (true);

drop policy if exists "public superclaims write" on public.taxonomy_superclaims;
create policy "public superclaims write"
  on public.taxonomy_superclaims for all
  using (true)
  with check (true);

drop policy if exists "public subclaims read" on public.taxonomy_subclaims;
create policy "public subclaims read"
  on public.taxonomy_subclaims for select
  using (true);

drop policy if exists "public subclaims write" on public.taxonomy_subclaims;
create policy "public subclaims write"
  on public.taxonomy_subclaims for all
  using (true)
  with check (true);

-- Allow PostgREST access for anon/authenticated roles.
grant usage on schema public to anon, authenticated;
grant select, insert, update, delete on public.taxonomy_superclaims to anon, authenticated;
grant select, insert, update, delete on public.taxonomy_subclaims to anon, authenticated;
