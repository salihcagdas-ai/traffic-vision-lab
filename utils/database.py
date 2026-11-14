"""Local, per-run SQLite records adapted from the graduation project."""

from pathlib import Path
import sqlite3


class TrafficDatabase:
    """Use a new database per video/run so tracker IDs cannot collide.

    By default, plate text, city, OCR confidence and evidence fields are
    discarded on write. This does not anonymize an existing database.
    """

    def __init__(self, db_path="outputs/violations.db", *, allow_personal_data=False):
        if not isinstance(allow_personal_data, bool):
            raise TypeError("allow_personal_data must be a boolean.")
        self.db_path = str(db_path)
        self.allow_personal_data = allow_personal_data
        if self.db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.cursor = self.conn.cursor()
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS vehicles (
                id INTEGER PRIMARY KEY,
                tracker_id INTEGER NOT NULL UNIQUE,
                plate_text TEXT, city TEXT, vehicle_class TEXT, confidence REAL,
                first_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS violations (
                id INTEGER PRIMARY KEY,
                vehicle_id INTEGER NOT NULL REFERENCES vehicles(id),
                violation_type INTEGER NOT NULL,
                speed REAL, speed_limit REAL, light_color TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                evidence_image TEXT, evidence_text TEXT
            );
        """)
        self.conn.commit()

    def add_vehicle(
        self, tracker_id, vehicle_class=None, plate_text=None, city=None, confidence=None
    ):
        if not self.allow_personal_data:
            plate_text = city = confidence = None
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO vehicles (tracker_id, vehicle_class, plate_text, city, confidence)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(tracker_id) DO UPDATE SET
                    vehicle_class = COALESCE(excluded.vehicle_class, vehicles.vehicle_class),
                    plate_text = COALESCE(vehicles.plate_text, excluded.plate_text),
                    city = COALESCE(vehicles.city, excluded.city),
                    confidence = COALESCE(vehicles.confidence, excluded.confidence),
                    last_seen = CURRENT_TIMESTAMP
            """,
                (int(tracker_id), vehicle_class, plate_text, city, confidence),
            )
        return self.conn.execute(
            "SELECT id FROM vehicles WHERE tracker_id = ?", (int(tracker_id),)
        ).fetchone()[0]

    def add_violation(
        self,
        vehicle_id,
        violation_type,
        speed=None,
        speed_limit=None,
        light_color=None,
        evidence_image=None,
        evidence_text=None,
    ):
        if not self.allow_personal_data:
            evidence_image = evidence_text = None
        with self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO violations
                (vehicle_id, violation_type, speed, speed_limit, light_color,
                 evidence_image, evidence_text) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    int(vehicle_id),
                    int(violation_type),
                    speed,
                    speed_limit,
                    light_color,
                    evidence_image,
                    evidence_text,
                ),
            )
        return cursor.lastrowid

    def get_vehicle_by_tracker_id(self, tracker_id):
        row = self.conn.execute(
            """
            SELECT id, tracker_id, plate_text, city, vehicle_class
            FROM vehicles WHERE tracker_id = ?
        """,
            (int(tracker_id),),
        ).fetchone()
        return dict(row) if row else None

    def get_vehicle_by_id(self, vehicle_id):
        row = self.conn.execute(
            """
            SELECT id, tracker_id, plate_text, city, vehicle_class
            FROM vehicles WHERE id = ?
        """,
            (int(vehicle_id),),
        ).fetchone()
        return dict(row) if row else None

    def get_violations_by_vehicle_id(self, vehicle_id):
        return [
            dict(row)
            for row in self.conn.execute(
                """
            SELECT id, violation_type, speed, speed_limit, light_color, timestamp,
                   evidence_image, evidence_text FROM violations
            WHERE vehicle_id = ? ORDER BY id DESC
        """,
                (int(vehicle_id),),
            )
        ]

    def get_all_violations(self, limit=100):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ValueError("limit must be a nonnegative integer.")
        return [
            dict(row)
            for row in self.conn.execute(
                """
            SELECT v.id, v.violation_type, v.speed, v.speed_limit, v.light_color,
                   v.timestamp, v.evidence_image, v.evidence_text,
                   vh.plate_text, vh.city, vh.vehicle_class
            FROM violations v JOIN vehicles vh ON v.vehicle_id = vh.id
            ORDER BY v.id DESC LIMIT ?
        """,
                (limit,),
            )
        ]

    def close(self):
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
