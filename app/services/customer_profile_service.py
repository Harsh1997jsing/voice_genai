"""
customer_profile_service.py — Business logic for managing customer profiles.

Each user can have multiple profiles but only ONE can be active at a time.
The active profile is used by the calling pipeline to load system_prompt / one_liner.
"""

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.customer_profile import CustomerProfile


# ─────────────────────────────────────────────────────────────────────────────
# Create
# ─────────────────────────────────────────────────────────────────────────────

def create_profile(
    db: Session,
    user_id: int,
    industry: str | None = None,
    description: str | None = None,
    system_prompt: str | None = None,
    set_active: bool = False,
) -> CustomerProfile:
    """Create a new customer profile. Optionally set it as active."""

    profile = CustomerProfile(
        user_id=user_id,
        industry=industry,
        description=description,
        system_prompt=system_prompt,
        is_active=False,
    )
    db.add(profile)
    db.flush()  # get the id before potential activate

    if set_active:
        _activate_profile(db, user_id, profile.id)

    db.commit()
    db.refresh(profile)
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────────────────────

def get_all_profiles(db: Session, user_id: int) -> list[CustomerProfile]:
    """Return all profiles for a user, ordered by most recent first."""
    return (
        db.query(CustomerProfile)
        .filter(CustomerProfile.user_id == user_id)
        .order_by(CustomerProfile.created_at.desc())
        .all()
    )


def get_profile_by_id(db: Session, user_id: int, profile_id: int) -> CustomerProfile:
    """Fetch a single profile, ensuring it belongs to the user."""
    profile = (
        db.query(CustomerProfile)
        .filter(
            CustomerProfile.id == profile_id,
            CustomerProfile.user_id == user_id,
        )
        .first()
    )
    if not profile:
        raise HTTPException(status_code=404, detail="Profile not found")
    return profile


def get_active_profile(db: Session, user_id: int) -> CustomerProfile | None:
    """Return the currently active profile for a user (or None)."""
    return (
        db.query(CustomerProfile)
        .filter(
            CustomerProfile.user_id == user_id,
            CustomerProfile.is_active == True,
        )
        .first()
    )


# ─────────────────────────────────────────────────────────────────────────────
# Update
# ─────────────────────────────────────────────────────────────────────────────

def update_profile(
    db: Session,
    user_id: int,
    profile_id: int,
    **fields,
) -> CustomerProfile:
    """Update editable fields on a profile."""

    profile = get_profile_by_id(db, user_id, profile_id)

    allowed_fields = {"industry", "description", "system_prompt"}

    for key, value in fields.items():
        if key in allowed_fields and value is not None:
            setattr(profile, key, value)

    db.commit()
    db.refresh(profile)
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Activate / Deactivate (on/off)
# ─────────────────────────────────────────────────────────────────────────────

def _activate_profile(db: Session, user_id: int, profile_id: int) -> None:
    """Internal helper — deactivate all, then activate one. No commit."""
    db.query(CustomerProfile).filter(
        CustomerProfile.user_id == user_id,
        CustomerProfile.is_active == True,
    ).update({"is_active": False})

    db.query(CustomerProfile).filter(
        CustomerProfile.id == profile_id,
        CustomerProfile.user_id == user_id,
    ).update({"is_active": True})


def activate_profile(db: Session, user_id: int, profile_id: int) -> CustomerProfile:
    """
    Set a profile as active. Deactivates all other profiles for this user.
    Only one profile can be active at a time.
    """
    profile = get_profile_by_id(db, user_id, profile_id)
    _activate_profile(db, user_id, profile.id)
    db.commit()
    db.refresh(profile)
    return profile


def deactivate_profile(db: Session, user_id: int, profile_id: int) -> CustomerProfile:
    """Turn off a profile (no profile will be active after this)."""
    profile = get_profile_by_id(db, user_id, profile_id)
    profile.is_active = False
    db.commit()
    db.refresh(profile)
    return profile


# ─────────────────────────────────────────────────────────────────────────────
# Delete
# ─────────────────────────────────────────────────────────────────────────────

def delete_profile(db: Session, user_id: int, profile_id: int) -> None:
    """Delete a profile. If it was active, no profile will be active."""
    profile = get_profile_by_id(db, user_id, profile_id)
    db.delete(profile)
    db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: serialize profile to dict (used by API layer)
# ─────────────────────────────────────────────────────────────────────────────

def profile_to_dict(profile: CustomerProfile) -> dict:
    """Convert a CustomerProfile ORM object to a JSON-safe dict."""
    return {
        "id": profile.id,
        "user_id": profile.user_id,
        "industry": profile.industry,
        "description": profile.description,
        "system_prompt": profile.system_prompt,
        "is_active": profile.is_active,
        "created_at": profile.created_at.isoformat() if profile.created_at else None,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }
