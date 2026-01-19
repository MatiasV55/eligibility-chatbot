from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from enum import Enum


class ConversationState(str, Enum):
    """Estados del flujo de conversación"""
    GREETING = "greeting"
    COLLECTING_PERSONAL_DATA = "collecting_personal_data"
    CONFIRMING_PERSONAL_DATA = "confirming_personal_data"
    COLLECTING_CAR_DATA = "collecting_car_data"
    CONFIRMING_CAR_DATA = "confirming_car_data"
    EVALUATING_ELIGIBILITY = "evaluating_eligibility"
    COMPLETED = "completed"


class PersonalData(BaseModel):
    """Datos personales del usuario"""
    full_name: Optional[str] = None
    birth_year: Optional[int] = None
    email: Optional[str] = None


class CarData(BaseModel):
    """Datos del vehículo"""
    brand: Optional[str] = None
    model: Optional[str] = None
    year: Optional[int] = None
    mileage: Optional[int] = None


class ChatState(BaseModel):
    """Estado completo de la conversación"""
    conversation_id: str
    current_step: ConversationState = ConversationState.GREETING
    personal_data: PersonalData = PersonalData()
    car_data: CarData = CarData()
    messages: list[dict] = []
    personal_data_confirmed: bool = False
    car_data_confirmed: bool = False
    eligibility_result: Optional[dict] = None
    created_at: datetime = datetime.now()
    updated_at: datetime = datetime.now()


class EligibilityResult(BaseModel):
    """Resultado de la evaluación de elegibilidad"""
    is_eligible: bool
    reasons: list[str]
    age: int
    car_age_ok: bool
    mileage_ok: bool
