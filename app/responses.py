"""
Respuestas pregeneradas para el chatbot según el flujo de conversación
"""
from typing import Optional, Dict


class ResponseGenerator:
    """
    Generador de respuestas pregeneradas para el chatbot.
    Por defecto usa respuestas predefinidas, pero puede usar LLM si está habilitado.
    """
    
    def __init__(self, use_llm_responses: bool = False, llm_provider=None):
        """
        Inicializa el generador de respuestas
        
        Args:
            use_llm_responses: Si True, usa LLM para generar respuestas personalizadas
            llm_provider: Proveedor LLM (solo necesario si use_llm_responses=True)
        """
        self.use_llm_responses = use_llm_responses
        self.llm_provider = llm_provider
    
    def _generate_with_llm(self, context: str, default_response: str) -> str:
        """
        Genera una respuesta usando LLM si está habilitado
        
        Args:
            context: Contexto para el LLM
            default_response: Respuesta por defecto si falla o no está habilitado
            
        Returns:
            Respuesta generada o por defecto
        """
        if not self.use_llm_responses or not self.llm_provider:
            return default_response
        
        try:
            from langchain_core.messages import HumanMessage
            
            prompt = f"""Eres un asistente virtual amable de KoolKars que ayuda a validar la elegibilidad de autos.
Genera una respuesta natural y amigable basada en el siguiente contexto:

{context}

IMPORTANTE: Mantén la respuesta breve, amigable y profesional. No agregues información extra.

Respuesta:"""
            
            response = self.llm_provider.invoke([HumanMessage(content=prompt)])
            generated = response.content.strip()
            return generated if generated else default_response
        except Exception:
            return default_response
    
    def greeting(self) -> str:
        """Respuesta inicial de saludo"""
        default = "¡Hola! Soy el asistente virtual de KoolKars. Estoy aquí para ayudarte a validar la elegibilidad de tu auto para nuestro producto. Antes de empezar, ¿podrías darme tu nombre completo?"
        context = "El usuario acaba de iniciar la conversación. Debes saludarlo, presentarte como asistente de KoolKars, explicar que ayudarás a validar la elegibilidad de su auto, y pedirle su nombre completo."
        return self._generate_with_llm(context, default)
    
    def ask_birth_year(self, first_name: Optional[str] = None) -> str:
        """Pregunta por el año de nacimiento"""
        if first_name:
            default = f"Gracias, {first_name}. Ahora, ¿cuál es tu año de nacimiento?"
            context = f"El usuario se llama {first_name}. Agradécele y pídele su año de nacimiento."
        else:
            default = "Ahora, ¿cuál es tu año de nacimiento?"
            context = "Pídele al usuario su año de nacimiento."
        return self._generate_with_llm(context, default)
    
    def ask_email(self) -> str:
        """Pregunta por el correo electrónico"""
        default = "Entendido. Finalmente, para enviarte el resumen de tu cotización, ¿cuál es tu dirección de correo electrónico?"
        context = "El usuario ya dio su año de nacimiento. Pídele su correo electrónico para enviarle el resumen de cotización."
        return self._generate_with_llm(context, default)
    
    def confirm_personal_data(self, personal_data: Dict) -> str:
        """Confirmación de datos personales"""
        name = personal_data.get("full_name", "")
        birth_year = personal_data.get("birth_year", "")
        email = personal_data.get("email", "")
        default = f"Perfecto! Entonces tengo: nombre {name}, año de nacimiento {birth_year} y email {email}. Esta todo correcto?"
        context = f"Confirma los datos personales: nombre={name}, año de nacimiento={birth_year}, email={email}. Pregunta si son correctos."
        return self._generate_with_llm(context, default)
    
    def personal_data_confirmed(self, first_name: str) -> str:
        """Respuesta cuando se confirman los datos personales"""
        default = f"Perfecto, {first_name}. Ya tengo tus datos personales. Ahora necesito información sobre tu vehículo. Para empezar, ¿cuál es la marca de tu auto? (Ejemplo: Toyota, Ford, Nissan)"
        context = f"El usuario {first_name} confirmó sus datos personales. Ahora debes pedirle información sobre su vehículo, empezando por la marca del auto."
        return self._generate_with_llm(context, default)
    
    def personal_data_reset(self) -> str:
        """Respuesta cuando se reinician los datos personales"""
        return "De acuerdo, empecemos de nuevo. ¿Cuál es tu nombre completo?"
    
    def _get_model_examples(self, brand: str) -> Optional[str]:
        """Obtiene ejemplos de modelos para una marca específica"""
        brand_models = {
            "toyota": "Corolla, Camry, RAV4",
            "honda": "Civic, CR-V, Accord",
            "ford": "Focus, Mustang, F-150",
            "chevrolet": "Cruze, Spark, Silverado",
            "nissan": "Sentra, Versa, Altima",
            "volkswagen": "Golf, Jetta, Tiguan",
            "hyundai": "Elantra, Tucson, Santa Fe",
            "kia": "Rio, Sportage, Sorento",
            "mazda": "Mazda3, CX-5, CX-30",
            "bmw": "Serie 3, X3, X5",
            "mercedes-benz": "Clase C, Clase E, GLC",
            "mercedes": "Clase C, Clase E, GLC",
            "audi": "A3, A4, Q5",
            "subaru": "Impreza, Outback, Forester",
            "jeep": "Wrangler, Cherokee, Grand Cherokee",
            "dodge": "Charger, Challenger, Durango",
            "ram": "1500, 2500, 3500",
            "fiat": "500, Cronos, Pulse",
            "renault": "Sandero, Duster, Kwid",
            "peugeot": "208, 308, 2008",
            "citroen": "C3, C4, Berlingo",
            "mitsubishi": "Lancer, Outlander, L200",
            "suzuki": "Swift, Vitara, Jimny",
            "lexus": "IS, ES, RX",
            "infiniti": "Q50, Q60, QX50",
            "acura": "TLX, MDX, RDX",
            "volvo": "S60, XC60, XC90",
            "land rover": "Range Rover, Discovery, Defender",
            "porsche": "911, Cayenne, Macan",
            "tesla": "Model 3, Model Y, Model S",
            "mini": "Cooper, Countryman, Clubman",
            "seat": "Ibiza, León, Arona",
        }
        return brand_models.get(brand.lower()) if brand else None
    
    def ask_car_brand(self, brand: Optional[str] = None) -> str:
        """Pregunta por el modelo del auto después de obtener la marca"""
        if brand:
            examples = self._get_model_examples(brand)
            if examples:
                return f"¿Y cuál es el modelo exacto de tu {brand}? (Ejemplo: {examples})"
            return f"¿Y cuál es el modelo exacto de tu {brand}?"
        return "¿Y cuál es el modelo exacto de tu auto?"
    
    def ask_car_model(self, brand: str, model: Optional[str] = None) -> str:
        """Pregunta por el modelo del auto"""
        if model:
            return f"Excelente. ¿Qué año es tu {brand} {model}?"
        return f"Excelente. ¿Qué año es tu {brand}?"
    
    def ask_car_year(self) -> str:
        """Pregunta por el año del auto"""
        return "Por último, ¿cuál es el kilometraje aproximado actual de tu vehículo?"
    
    def confirm_car_data(self, car_data: Dict) -> str:
        """Confirmación de datos del auto"""
        brand = car_data.get("brand", "")
        model = car_data.get("model", "")
        year = car_data.get("year", "")
        mileage = car_data.get("mileage", 0)
        mileage_str = f"{mileage:,}".replace(',', '.')
        return f"Perfecto! {brand} {model} del {year} con {mileage_str}km, correcto?"
    
    def car_data_confirmed(self, first_name: str) -> str:
        """Respuesta cuando se confirman los datos del auto (no visible, va directo a evaluación)"""
        return ""  # No se muestra, va directo a evaluación
    
    def car_data_reset(self, first_name: str) -> str:
        """Respuesta cuando se reinician los datos del auto"""
        return f"De acuerdo, {first_name}. Empecemos de nuevo con los datos del auto. ¿Cuál es la marca de tu auto?"
    
    def eligibility_result(self, eligibility_result: Dict, first_name: str) -> str:
        """Mensaje de resultado de elegibilidad"""
        if not eligibility_result:
            return "Error al evaluar la elegibilidad."
        
        is_eligible = eligibility_result.get("is_eligible", False)
        
        if is_eligible:
            default = f"¡Buenas noticias, {first_name}! Basado en los criterios iniciales, eres elegible para nuestro producto!"
            context = f"El usuario {first_name} ES ELEGIBLE para el producto. Dale las buenas noticias de forma entusiasta."
            return self._generate_with_llm(context, default)
        else:
            reasons = eligibility_result.get("reasons", [])
            failed_reasons = [r for r in reasons if not r.startswith("✓")]
            default = f"Lamentablemente, {first_name}, no cumples con los criterios de elegibilidad:\n"
            for reason in failed_reasons:
                default += f"- {reason}\n"
            context = f"El usuario {first_name} NO ES ELEGIBLE. Razones: {', '.join(failed_reasons)}. Informa de forma empática."
            return self._generate_with_llm(context, default.strip())
    
    # Mensajes de error/validación
    def invalid_name(self) -> str:
        """Mensaje cuando no se puede extraer el nombre"""
        return "No pude entender tu nombre. ¿Podrías darme tu nombre completo? (Por ejemplo: Juan Pérez)"
    
    def invalid_birth_year(self) -> str:
        """Mensaje cuando no se puede extraer el año de nacimiento"""
        return "No pude entender el año. ¿Podrías darme tu año de nacimiento? (Ejemplo: 1995)"
    
    def invalid_email(self) -> str:
        """Mensaje cuando no se puede extraer el email"""
        return "No pude entender el email. ¿Podrías darme tu dirección de correo electrónico?"
    
    def invalid_car_brand(self) -> str:
        """Mensaje cuando no se puede extraer la marca del auto"""
        return "No pude entender la marca. ¿Podrías darme la marca de tu auto? (Ejemplo: Toyota, Ford, Honda)"
    
    def invalid_car_model(self, brand: Optional[str] = None) -> str:
        """Mensaje cuando no se puede extraer el modelo del auto"""
        if brand:
            examples = self._get_model_examples(brand)
            if examples:
                return f"No pude entender el modelo. ¿Podrías darme el modelo exacto de tu {brand}? (Ejemplo: {examples})"
            return f"No pude entender el modelo. ¿Podrías darme el modelo exacto de tu {brand}?"
        return "No pude entender el modelo. ¿Podrías darme el modelo exacto de tu auto?"
    
    def invalid_car_year(self) -> str:
        """Mensaje cuando no se puede extraer el año del auto"""
        return "No pude entender el año. ¿Podrías darme el año de tu vehículo? (Ejemplo: 2018)"
    
    def invalid_mileage(self) -> str:
        """Mensaje cuando no se puede extraer el kilometraje"""
        return "No pude entender el kilometraje. ¿Podrías darme el kilometraje de tu vehículo? (Ejemplo: 45000)"
    
    def invalid_confirmation(self) -> str:
        """Mensaje cuando la confirmación no es válida"""
        return "Por favor, responde con 'Sí' o 'No'. ¿Los datos son correctos?"
    
    def invalid_car_confirmation(self) -> str:
        """Mensaje cuando la confirmación del auto no es válida"""
        return "Por favor, responde con 'Sí' o 'No'. ¿Los datos del auto son correctos?"
