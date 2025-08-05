"""Translation and localization system for global deployment."""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import threading

logger = logging.getLogger(__name__)


@dataclass 
class Translation:
    """Represents a translation with metadata."""
    key: str
    language: str
    text: str
    context: Optional[str] = None
    plurals: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class Translator:
    """Multi-language translation system with fallback support."""
    
    SUPPORTED_LANGUAGES = {
        "en": {"name": "English", "native": "English", "rtl": False},
        "es": {"name": "Spanish", "native": "Español", "rtl": False},
        "fr": {"name": "French", "native": "Français", "rtl": False},
        "de": {"name": "German", "native": "Deutsch", "rtl": False},
        "ja": {"name": "Japanese", "native": "日本語", "rtl": False},
        "zh": {"name": "Chinese", "native": "中文", "rtl": False},
        "ar": {"name": "Arabic", "native": "العربية", "rtl": True},
        "ru": {"name": "Russian", "native": "Русский", "rtl": False},
        "pt": {"name": "Portuguese", "native": "Português", "rtl": False},
        "it": {"name": "Italian", "native": "Italiano", "rtl": False}
    }
    
    def __init__(self, translations_dir: str = "translations", default_language: str = "en"):
        self.translations_dir = Path(translations_dir)
        self.default_language = default_language
        self.current_language = default_language
        self.translations: Dict[str, Dict[str, Translation]] = {}
        self.fallback_chain = ["en"]  # Always fallback to English
        self._lock = threading.RLock()
        
        # Load translations
        self._load_translations()
    
    def _load_translations(self):
        """Load all translation files."""
        if not self.translations_dir.exists():
            logger.warning(f"Translations directory not found: {self.translations_dir}")
            self._create_default_translations()
            return
        
        for lang_code in self.SUPPORTED_LANGUAGES:
            lang_file = self.translations_dir / f"{lang_code}.json"
            if lang_file.exists():
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        self._load_language_data(lang_code, data)
                        logger.info(f"Loaded translations for {lang_code}")
                except Exception as e:
                    logger.error(f"Failed to load translations for {lang_code}: {e}")
            else:
                logger.warning(f"Translation file not found: {lang_file}")
    
    def _load_language_data(self, lang_code: str, data: Dict[str, Any]):
        """Load translation data for a specific language."""
        if lang_code not in self.translations:
            self.translations[lang_code] = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # Simple string translation
                translation = Translation(
                    key=key,
                    language=lang_code,
                    text=value
                )
            elif isinstance(value, dict):
                # Complex translation with plurals/context
                translation = Translation(
                    key=key,
                    language=lang_code,
                    text=value.get("text", ""),
                    context=value.get("context"),
                    plurals=value.get("plurals", {}),
                    metadata=value.get("metadata", {})
                )
            else:
                logger.warning(f"Invalid translation format for {key} in {lang_code}")
                continue
            
            self.translations[lang_code][key] = translation
    
    def _create_default_translations(self):
        """Create default English translations."""
        os.makedirs(self.translations_dir, exist_ok=True)
        
        default_translations = {
            # System messages
            "system.startup": "Healing Guard system starting up",
            "system.shutdown": "Healing Guard system shutting down",
            "system.ready": "System ready and operational",
            "system.error": "System error occurred",
            
            # API messages
            "api.validation_error": "Validation error in request",
            "api.authentication_required": "Authentication required",
            "api.access_denied": "Access denied",
            "api.resource_not_found": "Resource not found",
            "api.internal_error": "Internal server error",
            "api.rate_limit_exceeded": "Rate limit exceeded",
            
            # Healing messages
            "healing.plan_created": "Healing plan created successfully",
            "healing.plan_execution_started": "Healing plan execution started",
            "healing.plan_execution_completed": "Healing plan execution completed",
            "healing.plan_execution_failed": "Healing plan execution failed",
            "healing.action_successful": "Healing action completed successfully",
            "healing.action_failed": "Healing action failed",
            
            # Failure detection messages
            "failure.detected": "Pipeline failure detected",
            "failure.analysis_started": "Failure analysis started",
            "failure.analysis_completed": "Failure analysis completed",
            "failure.classification": "Failure classified as {type} with {confidence}% confidence",
            
            # Task planning messages
            "planning.started": "Task planning started",
            "planning.completed": "Task planning completed",
            "planning.optimization_running": "Running quantum optimization",
            "planning.execution_started": "Task execution started",
            "planning.execution_completed": "Task execution completed",
            
            # Security messages
            "security.scan_started": "Security scan started",
            "security.scan_completed": "Security scan completed",
            "security.vulnerability_found": "Security vulnerability found",
            "security.access_granted": "Access granted",
            "security.access_denied": "Access denied",
            
            # General UI labels
            "label.dashboard": "Dashboard",
            "label.failures": "Failures", 
            "label.healing": "Healing",
            "label.tasks": "Tasks",
            "label.security": "Security",
            "label.settings": "Settings",
            "label.status": "Status",
            "label.metrics": "Metrics",
            "label.logs": "Logs",
            "label.repository": "Repository",
            "label.branch": "Branch",
            "label.commit": "Commit",
            "label.duration": "Duration",
            "label.success_rate": "Success Rate",
            "label.error_rate": "Error Rate",
            
            # Action labels
            "action.retry": "Retry",
            "action.cancel": "Cancel",
            "action.save": "Save",
            "action.delete": "Delete",
            "action.edit": "Edit",
            "action.view": "View",
            "action.download": "Download",
            "action.export": "Export",
            "action.import": "Import",
            "action.refresh": "Refresh",
            
            # Status messages
            "status.pending": "Pending",
            "status.running": "Running",  
            "status.completed": "Completed",
            "status.failed": "Failed",
            "status.cancelled": "Cancelled",
            "status.healthy": "Healthy",
            "status.unhealthy": "Unhealthy",
            "status.degraded": "Degraded",
            
            # Time-related
            "time.seconds": {
                "text": "second",
                "plurals": {
                    "one": "second",
                    "other": "seconds"
                }
            },
            "time.minutes": {
                "text": "minute",
                "plurals": {
                    "one": "minute", 
                    "other": "minutes"
                }
            },
            "time.hours": {
                "text": "hour",
                "plurals": {
                    "one": "hour",
                    "other": "hours"
                }
            },
            "time.days": {
                "text": "day",
                "plurals": {
                    "one": "day",
                    "other": "days"
                }
            },
            
            # Error messages
            "error.network": "Network connection error",
            "error.timeout": "Request timeout",
            "error.parse": "Data parsing error",
            "error.validation": "Validation error",
            "error.permission": "Permission denied",
            "error.not_found": "Resource not found",
            "error.conflict": "Resource conflict",
            "error.server": "Server error",
            
            # Success messages
            "success.saved": "Successfully saved",
            "success.deleted": "Successfully deleted",
            "success.updated": "Successfully updated",
            "success.created": "Successfully created",
            "success.imported": "Successfully imported",
            "success.exported": "Successfully exported"
        }
        
        # Save default translations
        en_file = self.translations_dir / "en.json"
        with open(en_file, 'w', encoding='utf-8') as f:
            json.dump(default_translations, f, indent=2, ensure_ascii=False)
        
        # Load the created translations
        self._load_language_data("en", default_translations)
        logger.info("Created default English translations")
    
    def set_language(self, language: str):
        """Set the current language."""
        if language not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"Unsupported language: {language}, falling back to {self.default_language}")
            language = self.default_language
        
        with self._lock:
            self.current_language = language
            # Update fallback chain
            self.fallback_chain = [language]
            if language != "en":
                self.fallback_chain.append("en")
        
        logger.info(f"Language set to: {language}")
    
    def get_language(self) -> str:
        """Get the current language."""
        return self.current_language
    
    def get_supported_languages(self) -> Dict[str, Dict[str, Any]]:
        """Get list of supported languages."""
        return self.SUPPORTED_LANGUAGES.copy()
    
    def translate(
        self, 
        key: str, 
        language: Optional[str] = None,
        context: Optional[str] = None,
        count: Optional[int] = None,
        **kwargs
    ) -> str:
        """Translate a key to the specified or current language."""
        target_language = language or self.current_language
        
        with self._lock:
            # Try target language first, then fallback chain
            languages_to_try = [target_language] + [
                lang for lang in self.fallback_chain 
                if lang != target_language
            ]
            
            for lang in languages_to_try:
                if lang in self.translations and key in self.translations[lang]:
                    translation = self.translations[lang][key]
                    
                    # Handle pluralization
                    if count is not None and translation.plurals:
                        plural_key = self._get_plural_key(count, lang)
                        if plural_key in translation.plurals:
                            text = translation.plurals[plural_key]
                        else:
                            text = translation.text
                    else:
                        text = translation.text
                    
                    # Handle string formatting
                    if kwargs:
                        try:
                            text = text.format(**kwargs)
                        except (KeyError, ValueError) as e:
                            logger.warning(f"Failed to format translation {key}: {e}")
                    
                    return text
            
            # If no translation found, return the key itself
            logger.warning(f"No translation found for key: {key}")
            return key
    
    def _get_plural_key(self, count: int, language: str) -> str:
        """Get the appropriate plural key for a language and count."""
        # Simplified pluralization rules
        # In a production system, you'd use a proper pluralization library
        
        if language in ["ja", "zh"]:
            # Japanese and Chinese don't have plurals
            return "other"
        elif language in ["ru"]:
            # Russian has complex plural rules
            if count % 10 == 1 and count % 100 != 11:
                return "one"
            elif count % 10 in [2, 3, 4] and count % 100 not in [12, 13, 14]:
                return "few"
            else:
                return "many"
        elif language in ["ar"]:
            # Arabic has very complex plural rules
            if count == 0:
                return "zero"
            elif count == 1:
                return "one"
            elif count == 2:
                return "two"
            elif count <= 10:
                return "few"
            else:
                return "many"
        else:
            # Default English-style pluralization
            return "one" if count == 1 else "other"
    
    def t(self, key: str, **kwargs) -> str:
        """Shorthand for translate method."""
        return self.translate(key, **kwargs)
    
    def add_translation(self, language: str, key: str, text: str, context: Optional[str] = None):
        """Add a translation dynamically."""
        if language not in self.SUPPORTED_LANGUAGES:
            logger.warning(f"Adding translation for unsupported language: {language}")
        
        with self._lock:
            if language not in self.translations:
                self.translations[language] = {}
            
            self.translations[language][key] = Translation(
                key=key,
                language=language,
                text=text,
                context=context
            )
        
        logger.debug(f"Added translation: {language}.{key}")
    
    def get_translation_coverage(self) -> Dict[str, Dict[str, Any]]:
        """Get translation coverage statistics."""
        if "en" not in self.translations:
            return {}
        
        base_keys = set(self.translations["en"].keys())
        coverage = {}
        
        for lang_code in self.SUPPORTED_LANGUAGES:
            if lang_code in self.translations:
                lang_keys = set(self.translations[lang_code].keys())
                translated_count = len(lang_keys.intersection(base_keys))
                missing_count = len(base_keys - lang_keys)
                
                coverage[lang_code] = {
                    "total_keys": len(base_keys),
                    "translated_keys": translated_count,
                    "missing_keys": missing_count,
                    "coverage_percentage": (translated_count / len(base_keys)) * 100 if base_keys else 0,
                    "missing_key_list": list(base_keys - lang_keys)[:10]  # First 10 missing keys
                }
            else:
                coverage[lang_code] = {
                    "total_keys": len(base_keys),
                    "translated_keys": 0,
                    "missing_keys": len(base_keys),
                    "coverage_percentage": 0,
                    "missing_key_list": list(base_keys)[:10]
                }
        
        return coverage
    
    def export_translations(self, language: str, file_path: str):
        """Export translations for a language to a file."""
        if language not in self.translations:
            raise ValueError(f"No translations found for language: {language}")
        
        export_data = {}
        for key, translation in self.translations[language].items():
            if translation.plurals or translation.context:
                export_data[key] = {
                    "text": translation.text,
                    "context": translation.context,
                    "plurals": translation.plurals,
                    "metadata": translation.metadata
                }
            else:
                export_data[key] = translation.text
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Exported {len(export_data)} translations for {language} to {file_path}")
    
    def import_translations(self, language: str, file_path: str):
        """Import translations for a language from a file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self._load_language_data(language, data)
        logger.info(f"Imported translations for {language} from {file_path}")


# Global translator instance
_translator: Optional[Translator] = None
_translator_lock = threading.Lock()


def get_translator() -> Translator:
    """Get the global translator instance."""
    global _translator
    
    if _translator is None:
        with _translator_lock:
            if _translator is None:
                _translator = Translator()
    
    return _translator


def _(key: str, **kwargs) -> str:
    """Global translation function shorthand."""
    return get_translator().translate(key, **kwargs)