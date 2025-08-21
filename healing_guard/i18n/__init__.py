"""Internationalization (i18n) support for global deployment."""

from .translator import Translator, get_translator
# from .localization import LocalizationManager, get_localization_manager  # Future enhancement

__all__ = [
    "Translator",
    "get_translator", 
    "LocalizationManager",
    "get_localization_manager"
]