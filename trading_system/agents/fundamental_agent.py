"""
Fundamental/News Intelligence Agent
Continuously ingests real-time news, events, announcements and applies NLP-based sentiment analysis
"""

import requests
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
import json
from datetime import datetime, timedelta
import time
import re
from urllib.parse import urlencode

# Try to import NLP libraries
try:
    from transformers import pipeline
    import nltk
    from textblob import TextBlob
    HAS_NLP_LIBS = True
except ImportError:
    HAS_NLP_LIBS = False
    print("NLP libraries not installed, using basic text analysis")

logger = logging.getLogger(__name__)


class FundamentalIntelligenceAgent:
    """
    Continuously ingests real-time news, events, announcements and applies NLP-based sentiment analysis
    Translates news into market permission logic, not predictions
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.news_sources = config.get('news_sources', [])
        self.sentiment_threshold = config.get('sentiment_threshold', 0.3)
        self.risk_tolerance = config.get('risk_tolerance', 0.7)
        self.last_news_check = None
        
        # Initialize NLP components if available
        if HAS_NLP_LIBS:
            try:
                self.sentiment_analyzer = pipeline("sentiment-analysis", 
                                                 model="cardiffnlp/twitter-roberta-base-sentiment-latest")
            except:
                self.sentiment_analyzer = None
        else:
            self.sentiment_analyzer = None
        
        # Critical keywords for risk assessment
        self.positive_keywords = [
            'regulatory_approval', 'adoption', 'institutional', 'halving', 'upgrade', 
            'partnership', 'listing', 'positive', 'bullish', 'recovery', 'growth'
        ]
        
        self.negative_keywords = [
            'ban', 'crackdown', 'hack', 'theft', 'breach', 'negative', 'bearish', 
            'regulatory_concern', 'delisting', 'shut_down', 'fraud', 'collapse'
        ]
        
        self.neutral_keywords = [
            'update', 'news', 'report', 'analysis', 'commentary'
        ]
    
    def fetch_news(self, symbols: List[str] = ['BTC', 'ETH']) -> List[Dict]:
        """
        Fetch news from various sources
        """
        all_news = []
        
        # Fetch from CoinDesk API
        coindesk_news = self._fetch_coindesk_news()
        all_news.extend(coindesk_news)
        
        # Fetch from CryptoCompare API
        cryptocompare_news = self._fetch_cryptocompare_news(symbols)
        all_news.extend(cryptocompare_news)
        
        # Add any custom sources from config
        for source in self.news_sources:
            if 'cryptocompare' not in source.lower():
                custom_news = self._fetch_custom_news(source)
                all_news.extend(custom_news)
        
        return all_news
    
    def _fetch_coindesk_news(self) -> List[Dict]:
        """
        Fetch news from CoinDesk API
        """
        try:
            url = "https://api.coindesk.com/v1/bpi/currentprice.json"
            response = requests.get(url, timeout=10)
            
            # Note: CoinDesk API might have changed, so we'll use a general approach
            # This is just an example - actual implementation would depend on current API
            news_items = []
            
            # For demonstration purposes, creating mock news
            # In a real implementation, this would parse actual news
            mock_news = {
                'title': 'Crypto Market Update',
                'summary': 'General market sentiment and updates',
                'source': 'coindesk_mock',
                'timestamp': datetime.now(),
                'url': 'https://www.coindesk.com',
                'categories': ['general']
            }
            news_items.append(mock_news)
            
            return news_items
        except Exception as e:
            logger.error(f"Error fetching CoinDesk news: {e}")
            return []
    
    def _fetch_cryptocompare_news(self, symbols: List[str]) -> List[Dict]:
        """
        Fetch news from CryptoCompare API
        """
        try:
            # This is a placeholder - actual implementation would use the real API
            news_items = []
            
            # Mock implementation for demonstration
            for symbol in symbols:
                mock_news = {
                    'title': f'{symbol} Market Analysis',
                    'summary': f'Recent developments and sentiment around {symbol}',
                    'source': 'cryptocompare_mock',
                    'timestamp': datetime.now(),
                    'url': f'https://www.cryptocompare.com',
                    'categories': [symbol.lower()]
                }
                news_items.append(mock_news)
            
            return news_items
        except Exception as e:
            logger.error(f"Error fetching CryptoCompare news: {e}")
            return []
    
    def _fetch_custom_news(self, source_url: str) -> List[Dict]:
        """
        Fetch news from custom sources
        """
        try:
            response = requests.get(source_url, timeout=10)
            # Process the response based on the specific API
            # This is a placeholder implementation
            return []
        except Exception as e:
            logger.error(f"Error fetching from custom source {source_url}: {e}")
            return []
    
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze sentiment of news text using NLP techniques
        """
        if not text:
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'confidence': 0.5,
                'keywords_found': []
            }
        
        # Use transformer model if available
        if self.sentiment_analyzer and HAS_NLP_LIBS:
            try:
                result = self.sentiment_analyzer(text[:512])  # Limit text length for model
                sentiment_score = result[0]['score']
                
                if result[0]['label'] == 'POSITIVE':
                    label = 'positive'
                    sentiment_score = sentiment_score
                elif result[0]['label'] == 'NEGATIVE':
                    label = 'negative'
                    sentiment_score = -sentiment_score
                else:
                    label = 'neutral'
                    sentiment_score = 0.0
                
                return {
                    'sentiment_score': sentiment_score,
                    'sentiment_label': label,
                    'confidence': result[0]['score'],
                    'keywords_found': self._extract_keywords(text)
                }
            except Exception as e:
                logger.warning(f"Transformer sentiment analysis failed: {e}")
        
        # Fallback to TextBlob if transformer fails
        if HAS_NLP_LIBS:
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity  # Range: -1 to 1
                
                if polarity > 0.1:
                    label = 'positive'
                elif polarity < -0.1:
                    label = 'negative'
                else:
                    label = 'neutral'
                
                return {
                    'sentiment_score': polarity,
                    'sentiment_label': label,
                    'confidence': abs(polarity),
                    'keywords_found': self._extract_keywords(text)
                }
            except Exception as e:
                logger.warning(f"TextBlob sentiment analysis failed: {e}")
        
        # Basic keyword-based sentiment as ultimate fallback
        return self._basic_sentiment_analysis(text)
    
    def _basic_sentiment_analysis(self, text: str) -> Dict:
        """
        Basic sentiment analysis based on keyword matching
        """
        text_lower = text.lower()
        
        pos_matches = [kw for kw in self.positive_keywords if kw.replace('_', ' ') in text_lower]
        neg_matches = [kw for kw in self.negative_keywords if kw.replace('_', ' ') in text_lower]
        neu_matches = [kw for kw in self.neutral_keywords if kw.replace('_', ' ') in text_lower]
        
        # Calculate scores
        pos_score = len(pos_matches) * 0.5
        neg_score = len(neg_matches) * 0.5
        neu_score = len(neu_matches) * 0.1
        
        total_score = pos_score - neg_score
        normalized_score = np.clip(total_score / (len(pos_matches) + len(neg_matches) + 1), -1, 1)
        
        if normalized_score > 0.1:
            label = 'positive'
        elif normalized_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return {
            'sentiment_score': normalized_score,
            'sentiment_label': label,
            'confidence': min(1.0, (pos_score + neg_score) / 2.0),
            'keywords_found': pos_matches + neg_matches
        }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """
        Extract relevant keywords from text
        """
        text_lower = text.lower()
        found_keywords = []
        
        for kw in self.positive_keywords + self.negative_keywords:
            if kw.replace('_', ' ') in text_lower:
                found_keywords.append(kw)
        
        return found_keywords
    
    def detect_regulatory_risk(self, news_items: List[Dict]) -> Dict:
        """
        Detect regulatory risks from news
        """
        regulatory_keywords = [
            'regulation', 'regulatory', 'compliance', 'sec', 'cftc', 'fca', 'esma',
            'ban', 'prohibit', 'restrict', 'license', 'approval', 'violation', 'fine'
        ]
        
        regulatory_events = []
        risk_score = 0.0
        
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
            
            found_keywords = [kw for kw in regulatory_keywords if kw in text]
            
            if found_keywords:
                # Assess severity based on keywords
                severity = 0.0
                for kw in found_keywords:
                    if kw in ['ban', 'prohibit', 'crackdown']:
                        severity += 0.8
                    elif kw in ['regulation', 'regulatory', 'compliance']:
                        severity += 0.3
                    elif kw in ['fine', 'violation']:
                        severity += 0.5
                    else:
                        severity += 0.2
                
                event = {
                    'title': item.get('title'),
                    'source': item.get('source'),
                    'severity': severity,
                    'keywords': found_keywords,
                    'timestamp': item.get('timestamp')
                }
                
                regulatory_events.append(event)
                risk_score = max(risk_score, severity)
        
        return {
            'events': regulatory_events,
            'overall_risk_score': risk_score,
            'risk_level': self._determine_risk_level(risk_score)
        }
    
    def detect_security_incidents(self, news_items: List[Dict]) -> Dict:
        """
        Detect security incidents from news
        """
        security_keywords = [
            'hack', 'hacker', 'theft', 'breach', 'exploit', 'vulnerability', 'attack',
            'compromise', 'security', 'incident', 'phishing', 'malware', 'ddos'
        ]
        
        security_events = []
        risk_score = 0.0
        
        for item in news_items:
            text = f"{item.get('title', '')} {item.get('summary', '')}".lower()
            
            found_keywords = [kw for kw in security_keywords if kw in text]
            
            if found_keywords:
                # Assess impact based on keywords
                severity = 0.0
                for kw in found_keywords:
                    if kw in ['hack', 'theft', 'breach']:
                        severity += 0.9
                    elif kw in ['exploit', 'vulnerability']:
                        severity += 0.7
                    elif kw in ['attack', 'compromise']:
                        severity += 0.8
                    else:
                        severity += 0.4
                
                event = {
                    'title': item.get('title'),
                    'source': item.get('source'),
                    'severity': severity,
                    'keywords': found_keywords,
                    'timestamp': item.get('timestamp')
                }
                
                security_events.append(event)
                risk_score = max(risk_score, severity)
        
        return {
            'events': security_events,
            'overall_risk_score': risk_score,
            'risk_level': self._determine_risk_level(risk_score)
        }
    
    def _determine_risk_level(self, risk_score: float) -> str:
        """
        Determine risk level based on score
        """
        if risk_score >= 0.8:
            return 'critical'
        elif risk_score >= 0.6:
            return 'high'
        elif risk_score >= 0.4:
            return 'medium'
        elif risk_score >= 0.2:
            return 'low'
        else:
            return 'minimal'
    
    def analyze_market_permission(self, news_analysis: Dict) -> Dict:
        """
        Translate news analysis into market permission logic
        Determines if market conditions allow for trades
        """
        overall_sentiment = news_analysis['aggregate_sentiment']['average_sentiment']
        regulatory_risk = news_analysis['regulatory_risk']['overall_risk_score']
        security_risk = news_analysis['security_incidents']['overall_risk_score']
        
        # Market permission logic
        # Positive sentiment allows more trading
        sentiment_permission = 1.0 if overall_sentiment > -0.2 else 0.5 if overall_sentiment > -0.5 else 0.1
        
        # High regulatory risk reduces permission
        regulatory_permission = 1.0 if regulatory_risk < 0.3 else 0.5 if regulatory_risk < 0.6 else 0.1
        
        # High security risk reduces permission  
        security_permission = 1.0 if security_risk < 0.3 else 0.5 if security_risk < 0.6 else 0.1
        
        # Overall market permission
        market_permission = min(sentiment_permission, regulatory_permission, security_permission)
        
        # Determine if trading should be restricted
        if regulatory_risk > 0.7 or security_risk > 0.7:
            trading_allowed = 'STRONG_RESTRICTION'
            reasoning = "High regulatory or security risk detected, severe trading restrictions applied"
        elif regulatory_risk > 0.5 or security_risk > 0.5 or overall_sentiment < -0.5:
            trading_allowed = 'MODERATE_RESTRICTION'
            reasoning = "Moderate risk detected, trading restrictions applied"
        elif overall_sentiment > 0.3:
            trading_allowed = 'FULL_PERMISSION'
            reasoning = "Positive sentiment and low risk environment, full trading allowed"
        else:
            trading_allowed = 'CAUTIOUS_PERMISSION'
            reasoning = "Neutral sentiment with manageable risk, cautious trading allowed"
        
        return {
            'market_permission_level': trading_allowed,
            'permission_score': market_permission,
            'trading_restriction_reasoning': reasoning,
            'sentiment_factor': sentiment_permission,
            'regulatory_factor': regulatory_permission,
            'security_factor': security_permission
        }
    
    def analyze_news_intelligence(self, symbols: List[str] = ['BTC', 'ETH']) -> Dict:
        """
        Main method to analyze news and intelligence
        """
        # Fetch latest news
        news_items = self.fetch_news(symbols)
        
        if not news_items:
            return {
                'timestamp': datetime.now(),
                'news_items_count': 0,
                'aggregate_sentiment': {
                    'average_sentiment': 0.0,
                    'positive_articles': 0,
                    'negative_articles': 0,
                    'neutral_articles': 0,
                    'most_negative_article': None,
                    'most_positive_article': None
                },
                'regulatory_risk': {
                    'events': [],
                    'overall_risk_score': 0.0,
                    'risk_level': 'minimal'
                },
                'security_incidents': {
                    'events': [],
                    'overall_risk_score': 0.0,
                    'risk_level': 'minimal'
                },
                'market_permission': {
                    'market_permission_level': 'CAUTIOUS_PERMISSION',
                    'permission_score': 0.6,
                    'trading_restriction_reasoning': "No recent news, maintaining cautious stance"
                },
                'recommendation': 'HOLD'
            }
        
        # Analyze sentiment for each article
        sentiments = []
        for item in news_items:
            sentiment = self.analyze_sentiment(f"{item.get('title', '')} {item.get('summary', '')}")
            item['sentiment'] = sentiment
            sentiments.append(sentiment['sentiment_score'])
        
        # Aggregate sentiment analysis
        avg_sentiment = np.mean(sentiments) if sentiments else 0.0
        positive_count = sum(1 for s in sentiments if s > 0.1)
        negative_count = sum(1 for s in sentiments if s < -0.1)
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Find most extreme articles
        if sentiments:
            most_positive_idx = np.argmax(sentiments)
            most_negative_idx = np.argmin(sentiments)
            most_positive_article = news_items[most_positive_idx]
            most_negative_article = news_items[most_negative_idx]
        else:
            most_positive_article = None
            most_negative_article = None
        
        # Detect regulatory risks
        regulatory_risk = self.detect_regulatory_risk(news_items)
        
        # Detect security incidents
        security_incidents = self.detect_security_incidents(news_items)
        
        # Determine market permission
        market_permission = self.analyze_market_permission({
            'aggregate_sentiment': {
                'average_sentiment': avg_sentiment,
                'positive_articles': positive_count,
                'negative_articles': negative_count,
                'neutral_articles': neutral_count,
                'most_negative_article': most_negative_article,
                'most_positive_article': most_positive_article
            },
            'regulatory_risk': regulatory_risk,
            'security_incidents': security_incidents
        })
        
        # Generate recommendation based on analysis
        if market_permission['permission_score'] < 0.3:
            recommendation = 'HOLD'
        elif market_permission['permission_score'] > 0.7 and avg_sentiment > 0.3:
            recommendation = 'BUY_ALLOWED'
        elif market_permission['permission_score'] > 0.7 and avg_sentiment < -0.3:
            recommendation = 'SELL_ALLOWED'
        else:
            recommendation = 'CAUTIOUS'
        
        return {
            'timestamp': datetime.now(),
            'news_items_count': len(news_items),
            'news_items': news_items,
            'aggregate_sentiment': {
                'average_sentiment': avg_sentiment,
                'positive_articles': positive_count,
                'negative_articles': negative_count,
                'neutral_articles': neutral_count,
                'most_negative_article': most_negative_article,
                'most_positive_article': most_positive_article
            },
            'regulatory_risk': regulatory_risk,
            'security_incidents': security_incidents,
            'market_permission': market_permission,
            'recommendation': recommendation
        }
    
    def generate_signal(self, news_analysis: Dict) -> Dict:
        """
        Generate buy/sell/hold signal based on news analysis
        """
        market_permission = news_analysis['market_permission']['permission_score']
        avg_sentiment = news_analysis['aggregate_sentiment']['average_sentiment']
        regulatory_risk = news_analysis['regulatory_risk']['overall_risk_score']
        security_risk = news_analysis['security_incidents']['overall_risk_score']
        
        # If high risk is detected, restrict trading regardless of sentiment
        if regulatory_risk > 0.7 or security_risk > 0.7:
            action = 'HOLD'
            confidence = 0.9
            reasoning = "High regulatory or security risk detected, trading restricted"
        elif market_permission < 0.3:
            action = 'HOLD'
            confidence = 0.8
            reasoning = "Market conditions not favorable due to negative sentiment or high risk"
        elif avg_sentiment > 0.4 and regulatory_risk < 0.5 and security_risk < 0.5:
            action = 'BUY'
            confidence = min(0.9, market_permission)
            reasoning = "Positive sentiment with manageable risk, buy signal generated"
        elif avg_sentiment < -0.4 and regulatory_risk < 0.5 and security_risk < 0.5:
            action = 'SELL'
            confidence = min(0.9, market_permission)
            reasoning = "Negative sentiment with manageable risk, sell signal generated"
        else:
            action = 'HOLD'
            confidence = max(0.3, market_permission * 0.7)
            reasoning = "Uncertain market conditions or mixed sentiment, hold position"
        
        return {
            'action': action,
            'confidence': confidence,
            'reasoning': reasoning,
            'risk_factors': {
                'sentiment_risk': abs(avg_sentiment) > 0.5,
                'regulatory_risk': regulatory_risk > 0.5,
                'security_risk': security_risk > 0.5
            }
        }


def main():
    """Test the Fundamental Intelligence Agent"""
    config = {
        'news_sources': [
            'https://min-api.cryptocompare.com/data/v2/news/',
        ],
        'sentiment_threshold': 0.3,
        'risk_tolerance': 0.7
    }
    
    agent = FundamentalIntelligenceAgent(config)
    
    # Test with some example symbols
    symbols = ['BTC', 'ETH']
    analysis = agent.analyze_news_intelligence(symbols)
    
    print("Fundamental Analysis Results:")
    print(f"News Items Count: {analysis['news_items_count']}")
    print(f"Average Sentiment: {analysis['aggregate_sentiment']['average_sentiment']:.2f}")
    print(f"Market Permission Score: {analysis['market_permission']['permission_score']:.2f}")
    print(f"Recommendation: {analysis['recommendation']}")
    
    signal = agent.generate_signal(analysis)
    print(f"\nSignal: {signal['action']}")
    print(f"Confidence: {signal['confidence']:.2f}")
    print(f"Reasoning: {signal['reasoning']}")


if __name__ == "__main__":
    main()