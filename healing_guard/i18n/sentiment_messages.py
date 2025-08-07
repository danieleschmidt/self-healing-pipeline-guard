"""Internationalization support for sentiment analysis messages and responses."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum

from ..core.sentiment_analyzer import SentimentLabel


class SupportedLanguage(Enum):
    """Supported languages for internationalization."""
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    JAPANESE = "ja"
    CHINESE = "zh"
    PORTUGUESE = "pt"
    ITALIAN = "it"


@dataclass
class LocalizedMessage:
    """Localized message with language support."""
    key: str
    language: SupportedLanguage
    message: str
    description: Optional[str] = None


class SentimentMessageLocalizer:
    """Localizes sentiment analysis messages for different languages."""
    
    def __init__(self):
        self.messages = self._initialize_messages()
    
    def _initialize_messages(self) -> Dict[str, Dict[SupportedLanguage, str]]:
        """Initialize localized messages for sentiment analysis."""
        return {
            # Sentiment labels
            "sentiment.very_positive": {
                SupportedLanguage.ENGLISH: "Very Positive",
                SupportedLanguage.SPANISH: "Muy Positivo",
                SupportedLanguage.FRENCH: "Très Positif",
                SupportedLanguage.GERMAN: "Sehr Positiv",
                SupportedLanguage.JAPANESE: "とても前向き",
                SupportedLanguage.CHINESE: "非常积极",
                SupportedLanguage.PORTUGUESE: "Muito Positivo",
                SupportedLanguage.ITALIAN: "Molto Positivo"
            },
            "sentiment.positive": {
                SupportedLanguage.ENGLISH: "Positive",
                SupportedLanguage.SPANISH: "Positivo",
                SupportedLanguage.FRENCH: "Positif",
                SupportedLanguage.GERMAN: "Positiv",
                SupportedLanguage.JAPANESE: "前向き",
                SupportedLanguage.CHINESE: "积极",
                SupportedLanguage.PORTUGUESE: "Positivo",
                SupportedLanguage.ITALIAN: "Positivo"
            },
            "sentiment.neutral": {
                SupportedLanguage.ENGLISH: "Neutral",
                SupportedLanguage.SPANISH: "Neutro",
                SupportedLanguage.FRENCH: "Neutre",
                SupportedLanguage.GERMAN: "Neutral",
                SupportedLanguage.JAPANESE: "中立",
                SupportedLanguage.CHINESE: "中性",
                SupportedLanguage.PORTUGUESE: "Neutro",
                SupportedLanguage.ITALIAN: "Neutro"
            },
            "sentiment.negative": {
                SupportedLanguage.ENGLISH: "Negative",
                SupportedLanguage.SPANISH: "Negativo",
                SupportedLanguage.FRENCH: "Négatif",
                SupportedLanguage.GERMAN: "Negativ",
                SupportedLanguage.JAPANESE: "否定的",
                SupportedLanguage.CHINESE: "消极",
                SupportedLanguage.PORTUGUESE: "Negativo",
                SupportedLanguage.ITALIAN: "Negativo"
            },
            "sentiment.very_negative": {
                SupportedLanguage.ENGLISH: "Very Negative",
                SupportedLanguage.SPANISH: "Muy Negativo",
                SupportedLanguage.FRENCH: "Très Négatif",
                SupportedLanguage.GERMAN: "Sehr Negativ",
                SupportedLanguage.JAPANESE: "とても否定的",
                SupportedLanguage.CHINESE: "非常消极",
                SupportedLanguage.PORTUGUESE: "Muito Negativo",
                SupportedLanguage.ITALIAN: "Molto Negativo"
            },
            "sentiment.urgent": {
                SupportedLanguage.ENGLISH: "Urgent",
                SupportedLanguage.SPANISH: "Urgente",
                SupportedLanguage.FRENCH: "Urgent",
                SupportedLanguage.GERMAN: "Dringend",
                SupportedLanguage.JAPANESE: "緊急",
                SupportedLanguage.CHINESE: "紧急",
                SupportedLanguage.PORTUGUESE: "Urgente",
                SupportedLanguage.ITALIAN: "Urgente"
            },
            "sentiment.frustrated": {
                SupportedLanguage.ENGLISH: "Frustrated",
                SupportedLanguage.SPANISH: "Frustrado",
                SupportedLanguage.FRENCH: "Frustré",
                SupportedLanguage.GERMAN: "Frustriert",
                SupportedLanguage.JAPANESE: "イライラ",
                SupportedLanguage.CHINESE: "沮丧",
                SupportedLanguage.PORTUGUESE: "Frustrado",
                SupportedLanguage.ITALIAN: "Frustrato"
            },
            "sentiment.confident": {
                SupportedLanguage.ENGLISH: "Confident",
                SupportedLanguage.SPANISH: "Confiado",
                SupportedLanguage.FRENCH: "Confiant",
                SupportedLanguage.GERMAN: "Zuversichtlich",
                SupportedLanguage.JAPANESE: "自信",
                SupportedLanguage.CHINESE: "自信",
                SupportedLanguage.PORTUGUESE: "Confiante",
                SupportedLanguage.ITALIAN: "Fiducioso"
            },
            
            # Analysis results
            "analysis.high_confidence": {
                SupportedLanguage.ENGLISH: "High confidence analysis",
                SupportedLanguage.SPANISH: "Análisis de alta confianza",
                SupportedLanguage.FRENCH: "Analyse à haute confiance",
                SupportedLanguage.GERMAN: "Analyse mit hoher Sicherheit",
                SupportedLanguage.JAPANESE: "高い信頼度の分析",
                SupportedLanguage.CHINESE: "高置信度分析",
                SupportedLanguage.PORTUGUESE: "Análise de alta confiança",
                SupportedLanguage.ITALIAN: "Analisi ad alta confidenza"
            },
            "analysis.low_confidence": {
                SupportedLanguage.ENGLISH: "Low confidence analysis",
                SupportedLanguage.SPANISH: "Análisis de baja confianza",
                SupportedLanguage.FRENCH: "Analyse à faible confiance",
                SupportedLanguage.GERMAN: "Analyse mit niedriger Sicherheit",
                SupportedLanguage.JAPANESE: "低い信頼度の分析",
                SupportedLanguage.CHINESE: "低置信度分析",
                SupportedLanguage.PORTUGUESE: "Análise de baixa confiança",
                SupportedLanguage.ITALIAN: "Analisi a bassa confidenza"
            },
            "analysis.completed": {
                SupportedLanguage.ENGLISH: "Sentiment analysis completed",
                SupportedLanguage.SPANISH: "Análisis de sentimientos completado",
                SupportedLanguage.FRENCH: "Analyse de sentiment terminée",
                SupportedLanguage.GERMAN: "Sentimentanalyse abgeschlossen",
                SupportedLanguage.JAPANESE: "感情分析完了",
                SupportedLanguage.CHINESE: "情感分析完成",
                SupportedLanguage.PORTUGUESE: "Análise de sentimentos concluída",
                SupportedLanguage.ITALIAN: "Analisi del sentimento completata"
            },
            
            # Urgency indicators
            "urgency.immediate_attention": {
                SupportedLanguage.ENGLISH: "Immediate attention required",
                SupportedLanguage.SPANISH: "Se requiere atención inmediata",
                SupportedLanguage.FRENCH: "Attention immédiate requise",
                SupportedLanguage.GERMAN: "Sofortige Aufmerksamkeit erforderlich",
                SupportedLanguage.JAPANESE: "即座の対応が必要",
                SupportedLanguage.CHINESE: "需要立即关注",
                SupportedLanguage.PORTUGUESE: "Atenção imediata necessária",
                SupportedLanguage.ITALIAN: "Attenzione immediata richiesta"
            },
            "urgency.production_impact": {
                SupportedLanguage.ENGLISH: "Production environment impacted",
                SupportedLanguage.SPANISH: "Entorno de producción afectado",
                SupportedLanguage.FRENCH: "Environnement de production impacté",
                SupportedLanguage.GERMAN: "Produktionsumgebung betroffen",
                SupportedLanguage.JAPANESE: "本番環境への影響",
                SupportedLanguage.CHINESE: "生产环境受到影响",
                SupportedLanguage.PORTUGUESE: "Ambiente de produção impactado",
                SupportedLanguage.ITALIAN: "Ambiente di produzione impattato"
            },
            
            # Healing recommendations
            "healing.priority_increased": {
                SupportedLanguage.ENGLISH: "Healing priority increased due to sentiment analysis",
                SupportedLanguage.SPANISH: "Prioridad de sanación aumentada debido al análisis de sentimientos",
                SupportedLanguage.FRENCH: "Priorité de guérison augmentée grâce à l'analyse de sentiment",
                SupportedLanguage.GERMAN: "Heilungspriorität durch Sentimentanalyse erhöht",
                SupportedLanguage.JAPANESE: "感情分析により治癒優先度が上がりました",
                SupportedLanguage.CHINESE: "由于情感分析，治愈优先级增加",
                SupportedLanguage.PORTUGUESE: "Prioridade de cura aumentada devido à análise de sentimentos",
                SupportedLanguage.ITALIAN: "Priorità di guarigione aumentata a causa dell'analisi del sentimento"
            },
            "healing.enhanced_actions": {
                SupportedLanguage.ENGLISH: "Healing actions enhanced based on emotional context",
                SupportedLanguage.SPANISH: "Acciones de sanación mejoradas basadas en el contexto emocional",
                SupportedLanguage.FRENCH: "Actions de guérison améliorées basées sur le contexte émotionnel",
                SupportedLanguage.GERMAN: "Heilungsaktionen basierend auf emotionalem Kontext verbessert",
                SupportedLanguage.JAPANESE: "感情的な文脈に基づいて治癒行動が強化されました",
                SupportedLanguage.CHINESE: "基于情感背景增强治愈行动",
                SupportedLanguage.PORTUGUESE: "Ações de cura aprimoradas com base no contexto emocional",
                SupportedLanguage.ITALIAN: "Azioni di guarigione migliorate basate sul contesto emotivo"
            },
            
            # Error messages
            "error.analysis_failed": {
                SupportedLanguage.ENGLISH: "Sentiment analysis failed",
                SupportedLanguage.SPANISH: "El análisis de sentimientos falló",
                SupportedLanguage.FRENCH: "L'analyse de sentiment a échoué",
                SupportedLanguage.GERMAN: "Sentimentanalyse fehlgeschlagen",
                SupportedLanguage.JAPANESE: "感情分析が失敗しました",
                SupportedLanguage.CHINESE: "情感分析失败",
                SupportedLanguage.PORTUGUESE: "Análise de sentimentos falhou",
                SupportedLanguage.ITALIAN: "Analisi del sentimento fallita"
            },
            "error.invalid_input": {
                SupportedLanguage.ENGLISH: "Invalid input provided",
                SupportedLanguage.SPANISH: "Entrada inválida proporcionada",
                SupportedLanguage.FRENCH: "Entrée invalide fournie",
                SupportedLanguage.GERMAN: "Ungültige Eingabe bereitgestellt",
                SupportedLanguage.JAPANESE: "無効な入力が提供されました",
                SupportedLanguage.CHINESE: "提供了无效输入",
                SupportedLanguage.PORTUGUESE: "Entrada inválida fornecida",
                SupportedLanguage.ITALIAN: "Input non valido fornito"
            },
            
            # Status messages
            "status.processing": {
                SupportedLanguage.ENGLISH: "Processing sentiment analysis...",
                SupportedLanguage.SPANISH: "Procesando análisis de sentimientos...",
                SupportedLanguage.FRENCH: "Traitement de l'analyse de sentiment...",
                SupportedLanguage.GERMAN: "Sentimentanalyse wird verarbeitet...",
                SupportedLanguage.JAPANESE: "感情分析を処理中...",
                SupportedLanguage.CHINESE: "正在处理情感分析...",
                SupportedLanguage.PORTUGUESE: "Processando análise de sentimentos...",
                SupportedLanguage.ITALIAN: "Elaborazione analisi del sentimento..."
            },
            "status.ready": {
                SupportedLanguage.ENGLISH: "System ready for sentiment analysis",
                SupportedLanguage.SPANISH: "Sistema listo para análisis de sentimientos",
                SupportedLanguage.FRENCH: "Système prêt pour l'analyse de sentiment",
                SupportedLanguage.GERMAN: "System bereit für Sentimentanalyse",
                SupportedLanguage.JAPANESE: "感情分析システム準備完了",
                SupportedLanguage.CHINESE: "系统准备进行情感分析",
                SupportedLanguage.PORTUGUESE: "Sistema pronto para análise de sentimentos",
                SupportedLanguage.ITALIAN: "Sistema pronto per analisi del sentimento"
            }
        }
    
    def get_message(
        self, 
        key: str, 
        language: SupportedLanguage = SupportedLanguage.ENGLISH,
        fallback_to_english: bool = True
    ) -> str:
        """Get localized message for a given key and language."""
        if key not in self.messages:
            return f"[Missing message: {key}]"
        
        message_dict = self.messages[key]
        
        # Try requested language first
        if language in message_dict:
            return message_dict[language]
        
        # Fallback to English if requested
        if fallback_to_english and SupportedLanguage.ENGLISH in message_dict:
            return message_dict[SupportedLanguage.ENGLISH]
        
        # Return first available message
        if message_dict:
            return next(iter(message_dict.values()))
        
        return f"[No translation available: {key}]"
    
    def get_sentiment_label(
        self, 
        sentiment_label: SentimentLabel,
        language: SupportedLanguage = SupportedLanguage.ENGLISH
    ) -> str:
        """Get localized sentiment label."""
        label_key = f"sentiment.{sentiment_label.value}"
        return self.get_message(label_key, language)
    
    def get_all_messages(self, language: SupportedLanguage) -> Dict[str, str]:
        """Get all messages for a specific language."""
        result = {}
        for key, message_dict in self.messages.items():
            result[key] = self.get_message(key, language)
        return result
    
    def is_language_supported(self, language_code: str) -> bool:
        """Check if a language code is supported."""
        try:
            SupportedLanguage(language_code)
            return True
        except ValueError:
            return False
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported language codes."""
        return [lang.value for lang in SupportedLanguage]
    
    def detect_language_from_accept_header(self, accept_language: str) -> SupportedLanguage:
        """Detect preferred language from HTTP Accept-Language header."""
        if not accept_language:
            return SupportedLanguage.ENGLISH
        
        # Parse Accept-Language header (simplified)
        # Format: "en-US,en;q=0.9,es;q=0.8"
        languages = []
        for part in accept_language.split(','):
            lang_part = part.split(';')[0].strip().lower()
            # Extract primary language code
            primary_lang = lang_part.split('-')[0]
            languages.append(primary_lang)
        
        # Check if any requested language is supported
        for lang_code in languages:
            if self.is_language_supported(lang_code):
                return SupportedLanguage(lang_code)
        
        return SupportedLanguage.ENGLISH


class LocalizedSentimentResponse:
    """Localized response wrapper for sentiment analysis results."""
    
    def __init__(self, localizer: SentimentMessageLocalizer):
        self.localizer = localizer
    
    def format_sentiment_result(
        self,
        sentiment_label: SentimentLabel,
        confidence: float,
        urgency_score: float,
        is_urgent: bool,
        is_frustrated: bool,
        language: SupportedLanguage = SupportedLanguage.ENGLISH
    ) -> Dict[str, Any]:
        """Format sentiment result with localized messages."""
        
        # Get localized sentiment label
        localized_label = self.localizer.get_sentiment_label(sentiment_label, language)
        
        # Determine confidence message
        confidence_key = "analysis.high_confidence" if confidence > 0.7 else "analysis.low_confidence"
        confidence_message = self.localizer.get_message(confidence_key, language)
        
        # Build response
        response = {
            "sentiment": {
                "label": sentiment_label.value,
                "localized_label": localized_label,
                "confidence": confidence,
                "confidence_message": confidence_message
            },
            "urgency": {
                "score": urgency_score,
                "is_urgent": is_urgent,
                "is_frustrated": is_frustrated
            },
            "messages": []
        }
        
        # Add urgency messages if applicable
        if is_urgent:
            response["messages"].append({
                "type": "urgency",
                "message": self.localizer.get_message("urgency.immediate_attention", language)
            })
        
        if is_frustrated:
            response["messages"].append({
                "type": "emotion",
                "message": self.localizer.get_message("sentiment.frustrated", language)
            })
        
        # Add completion message
        response["messages"].append({
            "type": "status",
            "message": self.localizer.get_message("analysis.completed", language)
        })
        
        return response
    
    def format_healing_enhancement_message(
        self,
        priority_increased: bool,
        actions_enhanced: bool,
        language: SupportedLanguage = SupportedLanguage.ENGLISH
    ) -> List[Dict[str, str]]:
        """Format healing enhancement messages."""
        messages = []
        
        if priority_increased:
            messages.append({
                "type": "healing_priority",
                "message": self.localizer.get_message("healing.priority_increased", language)
            })
        
        if actions_enhanced:
            messages.append({
                "type": "healing_enhancement",
                "message": self.localizer.get_message("healing.enhanced_actions", language)
            })
        
        return messages


# Global localizer instance
sentiment_localizer = SentimentMessageLocalizer()
localized_response_formatter = LocalizedSentimentResponse(sentiment_localizer)