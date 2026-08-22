# ================================================================
# MEDUSA AI
# REPORT SECURITY / REVIEW GUARD
#
# A report is considered downloadable ONLY when the exact
# scan_id has an APPROVED radiologist review.
# ================================================================

from utils.supabase_client import get_supabase


# ================================================================
# GET APPROVED REVIEW FOR EXACT SCAN
# ================================================================

def get_approved_review(scan_id):

    if not scan_id:
        return None

    supabase = get_supabase()

    response = (
        supabase
        .table("radiologist_reviews")
        .select("*")
        .eq("scan_id", scan_id)
        .eq("status", "APPROVED")
        .eq("approved", True)
        .order("reviewed_at", desc=True)
        .limit(1)
        .execute()
    )

    rows = response.data or []

    if not rows:
        return None

    return rows[0]


# ================================================================
# CHECK WHETHER EXACT SCAN IS APPROVED
# ================================================================

def has_approved_review(scan_id):

    review = get_approved_review(
        scan_id
    )

    return review is not None


# ================================================================
# REQUIRE APPROVAL
# ================================================================

def require_approved_review(scan_id):

    if not scan_id:

        raise PermissionError(
            "No scan ID was supplied."
        )

    if not has_approved_review(scan_id):

        raise PermissionError(
            "Report download is locked. "
            "This scan requires an approved "
            "radiologist review."
        )

    return True
