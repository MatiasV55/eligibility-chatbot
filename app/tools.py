from datetime import datetime
from langchain_core.tools import tool
from typing import Dict, Any
from app.models import EligibilityResult


@tool
def evaluate_eligibility(
    birth_year: int,
    car_year: int,
    mileage: int
) -> Dict[str, Any]:
    """
    Evalúa la elegibilidad de un cliente basándose en:
    - Edad: debe ser mayor de 18 años
    - Año del auto: no más viejo que 2015
    - Kilometraje: menor a 100,000 km
    
    Args:
        birth_year: Año de nacimiento del cliente
        car_year: Año del vehículo
        mileage: Kilometraje del vehículo
    
    Returns:
        Dict con el resultado de la evaluación
    """
    current_year = datetime.now().year
    age = current_year - birth_year
    
    # Criterios de elegibilidad
    age_ok = age >= 18
    car_age_ok = car_year >= 2015
    mileage_ok = mileage < 100000
    
    is_eligible = age_ok and car_age_ok and mileage_ok
    
    reasons = []
    if not age_ok:
        reasons.append(f"El cliente tiene {age} años, debe ser mayor de 18")
    else:
        reasons.append(f"✓ Edad: {age} años (mayor de 18)")
    
    if not car_age_ok:
        reasons.append(f"El auto es del año {car_year}, debe ser del 2015 o posterior")
    else:
        reasons.append(f"✓ Año del auto: {car_year} (posterior a 2015)")
    
    if not mileage_ok:
        reasons.append(f"El kilometraje es {mileage} km, debe ser menor a 100,000 km")
    else:
        reasons.append(f"✓ Kilometraje: {mileage} km (menor a 100,000 km)")
    
    result = EligibilityResult(
        is_eligible=is_eligible,
        reasons=reasons,
        age=age,
        car_age_ok=car_age_ok,
        mileage_ok=mileage_ok
    )
    
    return result.model_dump()
