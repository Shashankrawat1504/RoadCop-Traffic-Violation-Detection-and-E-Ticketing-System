"""Generate e-challan PDFs for traffic violations."""
from pathlib import Path
from fpdf import FPDF
FINE_INR = 1000

class _TicketPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "B", 14)
        self.cell(0, 10, "E-Challan (Traffic Violation)", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")


def write_eticket_pdf(out_path: str | Path, violation: dict, fine_inr: int = FINE_INR) -> None:
    """Write a single violation ticket as PDF (plate, violation, fine, time)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plate = violation.get("vehicle_number") or "UP"
    vtype = violation.get("violation_type") or "Unknown"
    ts = violation.get("timestamp") or ""
    passengers = violation.get("passenger_count")

    pdf = _TicketPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Vehicle / Plate Number: {plate}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Violation: {vtype}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Fine Amount: INR {fine_inr}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Date & Time: {ts}", new_x="LMARGIN", new_y="NEXT")
    if passengers is not None:
        pdf.cell(0, 8, f"Passengers (overload): {passengers}", new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(out_path))


def write_vehicle_ticket_pdf(out_path: str | Path, vehicle_number: str, violations: list[dict], fine_inr: int = FINE_INR) -> None:
    """Write one consolidated ticket per vehicle containing all violations."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plate = vehicle_number or "UNREAD"

    violation_types = {v.get("violation_type", "") for v in violations}
    has_no_helmet = "No Helmet" in violation_types
    has_overload = "Overloaded Vehicle" in violation_types
    # Rule requested: No Helmet + Triple Riding => INR 2000 total
    if has_no_helmet and has_overload:
        total_fine = 2000
    else:
        total_fine = fine_inr * len(violations)

    pdf = _TicketPDF()
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 8, f"Vehicle / Plate Number: {plate}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Total Violations: {len(violations)}", new_x="LMARGIN", new_y="NEXT")
    pdf.cell(0, 8, f"Total Fine: INR {total_fine}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 11)
    pdf.cell(0, 8, "Violation Details:", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 11)
    usable_w = max(40, pdf.w - pdf.l_margin - pdf.r_margin)

    for idx, v in enumerate(violations, start=1):
        vtype = v.get("violation_type", "Unknown")
        ts = v.get("timestamp", "")
        passengers = v.get("passenger_count")
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(usable_w, 7, f"{idx}. {vtype} | Fine: INR {fine_inr} | Time: {ts}")
        if passengers is not None:
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(usable_w, 7, f"   Passengers detected: {passengers}")
        pdf.ln(1)

    pdf.output(str(out_path))


def compute_total_fine_for_vehicle(violations: list[dict], fine_inr: int = FINE_INR) -> int:
    """Keep ticket math consistent for PDF and DB."""
    violation_types = {v.get("violation_type", "") for v in violations}
    has_no_helmet = "No Helmet" in violation_types
    has_overload = "Overloaded Vehicle" in violation_types
    if has_no_helmet and has_overload:
        return 2000
    return fine_inr * len(violations)


def violation_to_public_url(relative_path: str) -> str:
    return "/" + relative_path.replace("\\", "/").lstrip("/")
