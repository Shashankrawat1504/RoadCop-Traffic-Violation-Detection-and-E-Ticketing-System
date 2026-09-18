import json
import os
from datetime import datetime


def _is_mysql_enabled() -> bool:
    return os.environ.get("MYSQL_ENABLED", "1").strip().lower() in ("1", "true", "yes")


def _get_conn():
    import mysql.connector

    return mysql.connector.connect(
        host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("MYSQL_PORT", "3306")),
        user=os.environ.get("MYSQL_USER", "root"),
        password=os.environ.get("MYSQL_PASSWORD", ""),
        database=os.environ.get("MYSQL_DATABASE", "roadcop"),
        autocommit=True,
    )


def _ensure_table(conn):
    q = """
    CREATE TABLE IF NOT EXISTS etickets (
        id BIGINT AUTO_INCREMENT PRIMARY KEY,
        generated_at DATETIME NOT NULL,
        vehicle_number VARCHAR(64) NOT NULL,
        total_violations INT NOT NULL,
        total_fine INT NOT NULL,
        violation_summary TEXT NOT NULL,
        ticket_pdf_url TEXT NOT NULL,
        source_video VARCHAR(255) NULL
    )
    """
    cur = conn.cursor()
    cur.execute(q)
    cur.close()


def save_ticket_record(
    vehicle_number: str,
    violations: list[dict],
    total_fine: int,
    ticket_pdf_url: str,
    source_video: str | None = None,
) -> None:
    if not _is_mysql_enabled():
        return
    try:
        conn = _get_conn()
        _ensure_table(conn)
        summary = json.dumps(
            [
                {
                    "violation_type": v.get("violation_type"),
                    "timestamp": v.get("timestamp"),
                    "fine": v.get("fine"),
                    "passenger_count": v.get("passenger_count"),
                }
                for v in violations
            ],
            ensure_ascii=True,
        )
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO etickets
                (generated_at, vehicle_number, total_violations, total_fine, violation_summary, ticket_pdf_url, source_video)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                vehicle_number or "UNREAD",
                len(violations),
                int(total_fine),
                summary,
                ticket_pdf_url,
                source_video,
            ),
        )
        cur.close()
        conn.close()
    except Exception as e:
        # Keep pipeline alive even if MySQL is not reachable.
        msg = str(e)
        if "1045" in msg or "Access denied" in msg:
            if "using password: NO" in msg:
                print(
                    "[RoadCop] MySQL: connection rejected (no password sent). "
                    "Set MYSQL_PASSWORD in a .env file next to run.py, or set MYSQL_USER/MYSQL_PASSWORD in the environment."
                )
            else:
                print(
                    "[RoadCop] MySQL: access denied (wrong password or user not allowed). "
                    f"Check MYSQL_USER and MYSQL_PASSWORD. Detail: {e}"
                )
        else:
            print(f"[RoadCop] MySQL store failed: {e}")
