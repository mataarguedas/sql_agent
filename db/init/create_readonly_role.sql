-- Create (or refresh) the read-only role used by the agent at runtime.
-- Run under the admin role after loading northwind.sql.
--
-- The role gets SELECT on all current + future tables in the `public` schema
-- and no other privileges. Combined with the code-level safety guard, this is
-- Layer 2 of the two-layer safety model (see CLAUDE.md §7).

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'sql_agent_ro') THEN
        CREATE ROLE sql_agent_ro LOGIN PASSWORD 'aAO1CdNhIR484fuYN3BFmPmL';
    END IF;
END
$$;

-- Revoke any pre-existing privileges, then grant only what's needed.
REVOKE ALL ON ALL TABLES    IN SCHEMA public FROM sql_agent_ro;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM sql_agent_ro;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA public FROM sql_agent_ro;
REVOKE ALL ON SCHEMA public                  FROM sql_agent_ro;

GRANT CONNECT ON DATABASE northwind TO sql_agent_ro;
GRANT USAGE   ON SCHEMA public      TO sql_agent_ro;
GRANT SELECT  ON ALL TABLES IN SCHEMA public TO sql_agent_ro;

-- Ensure the same holds for tables created later.
ALTER DEFAULT PRIVILEGES IN SCHEMA public
    GRANT SELECT ON TABLES TO sql_agent_ro;

-- Belt-and-braces: explicitly deny write/DDL at the role level.
-- NOSUPERUSER is omitted: managed Postgres providers (e.g. Neon) don't grant
-- their admin role real SUPERUSER, and Postgres refuses to let a non-superuser
-- touch the SUPERUSER attribute even as a no-op — CREATE ROLE ... LOGIN
-- already defaults to non-superuser, so nothing is lost.
ALTER ROLE sql_agent_ro NOCREATEDB NOCREATEROLE NOINHERIT NOREPLICATION;
