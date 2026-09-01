-- SQLite creation/seed script for the classroom mini project.

CREATE TABLE IF NOT EXISTS evaluation_criteria (
    criterion_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    weight REAL NOT NULL,
    max_score REAL NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS rfp_runs (
    rfp_run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS supplier_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    rfp_run_id TEXT NOT NULL,
    supplier_name TEXT NOT NULL,
    submission_date TEXT NOT NULL,
    experience_rating REAL NOT NULL,
    absolute_score REAL NOT NULL,
    ppi REAL NOT NULL,
    final_rank INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    FOREIGN KEY (rfp_run_id) REFERENCES rfp_runs(rfp_run_id)
);

INSERT OR IGNORE INTO evaluation_criteria
(criterion_id, name, description, weight, max_score, is_active) VALUES
(1, 'Technical Capability', 'Architecture, integrations, scalability, technical fit', 30, 10, 1),
(2, 'Implementation Plan', 'Timeline, milestones, staffing, risk plan', 20, 10, 1),
(3, 'Commercial Value', 'Pricing clarity, total cost, assumptions', 20, 10, 1),
(4, 'Security & Compliance', 'Controls, certifications, privacy, auditability', 20, 10, 1),
(5, 'Support & Experience', 'Support model, similar projects, references', 10, 10, 1);
