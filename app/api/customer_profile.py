from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.deps import get_current_user
from app.db.database import get_db
from app.models.user import User
from app.services.customer_profile_service import (
    activate_profile,
    create_profile,
    deactivate_profile,
    delete_profile,
    get_active_profile,
    get_all_profiles,
    get_profile_by_id,
    profile_to_dict,
    update_profile,
)

router = APIRouter(prefix="/profile", tags=["Customer Profiles"])


# ─────────────────────────────────────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────────────────────────────────────

class CreateProfileRequest(BaseModel):
    industry: str | None = None
    description: str | None = None
    system_prompt: str | None = None
    set_active: bool = False


class UpdateProfileRequest(BaseModel):
    industry: str | None = None
    description: str | None = None
    system_prompt: str | None = None


# ─────────────────────────────────────────────────────────────────────────────
# POST  /profile          → create new profile
# ─────────────────────────────────────────────────────────────────────────────

@router.post("")
async def create_customer_profile(
    body: CreateProfileRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new customer profile. Optionally set it as the active one."""
    profile = create_profile(
        db=db,
        user_id=current_user.id,
        industry=body.industry,
        description=body.description,
        system_prompt=body.system_prompt,
        set_active=body.set_active,
    )
    return {"message": "Profile created", "profile": profile_to_dict(profile)}


# ─────────────────────────────────────────────────────────────────────────────
# GET   /profile          → list all profiles
# GET   /profile/active   → get the currently active profile
# GET   /profile/{id}     → get single profile
# ─────────────────────────────────────────────────────────────────────────────

@router.get("")
async def list_profiles(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all customer profiles for the current user."""
    profiles = get_all_profiles(db, current_user.id)
    return {
        "total": len(profiles),
        "profiles": [profile_to_dict(p) for p in profiles],
    }


@router.get("/active")
async def get_active(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get the currently active profile (if any)."""
    profile = get_active_profile(db, current_user.id)
    if not profile:
        return {"message": "No active profile", "profile": None}
    return {"profile": profile_to_dict(profile)}


@router.get("/{profile_id}")
async def get_single_profile(
    profile_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single profile by ID."""
    profile = get_profile_by_id(db, current_user.id, profile_id)
    return {"profile": profile_to_dict(profile)}


# ─────────────────────────────────────────────────────────────────────────────
# PUT   /profile/{id}     → update profile fields
# ─────────────────────────────────────────────────────────────────────────────

@router.put("/{profile_id}")
async def update_customer_profile(
    profile_id: int,
    body: UpdateProfileRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Update editable fields on a profile."""
    profile = update_profile(
        db=db,
        user_id=current_user.id,
        profile_id=profile_id,
        **body.model_dump(exclude_none=True),
    )
    return {"message": "Profile updated", "profile": profile_to_dict(profile)}


# ─────────────────────────────────────────────────────────────────────────────
# POST  /profile/{id}/activate     → turn ON  (sets this as the active one)
# POST  /profile/{id}/deactivate   → turn OFF
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/{profile_id}/activate")
async def activate(
    profile_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Activate this profile for calls.
    Automatically deactivates any other active profile.
    """
    profile = activate_profile(db, current_user.id, profile_id)
    return {"message": f"Profile {profile.id} is now active", "profile": profile_to_dict(profile)}


@router.post("/{profile_id}/deactivate")
async def deactivate(
    profile_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Deactivate this profile. No profile will be active after this."""
    profile = deactivate_profile(db, current_user.id, profile_id)
    return {"message": f"Profile {profile.id} deactivated", "profile": profile_to_dict(profile)}


# ─────────────────────────────────────────────────────────────────────────────
# DELETE /profile/{id}    → delete profile
# ─────────────────────────────────────────────────────────────────────────────

@router.delete("/{profile_id}")
async def delete_customer_profile(
    profile_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Delete a profile permanently."""
    delete_profile(db, current_user.id, profile_id)
    return {"message": f"Profile {profile_id} deleted"}
